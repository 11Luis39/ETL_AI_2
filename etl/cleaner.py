"""Reglas de limpieza y registro de propiedades excluidas."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


TIPO_PROPIEDAD_MAP = {
    "Casa": "Casa",
    "Casa de Calidad": "Casa",
    "Casa de Campo": "Casa",
    "Casa con Espacio Comercial": "Casa",
    "Departamento": "Departamento",
    "Dúplex": "Departamento",
    "Penthouse": "Departamento",
    "Estudio/Monoambiente": "Departamento",
    "Condominio / Departamento": "Departamento",
    "Apartamento con servicio de hotel": "Departamento",
    "Local Comercial": "Local Comercial",
    "Comercial/Negocio": "Local Comercial",
    "Oficina": "Oficina",
    "Baulera": "Otro",
    "Clínica de salud": "Otro",
    "Edificio": "Otro",
    "Edificio de apartamentos entero": "Otro",
    "Edificio/Construcción": "Otro",
    "Galpon": "Otro",
    "Garaje/Baulera": "Otro",
    "Hotel/Edificio de apartamentos": "Otro",
    "Quinta": "Otro",
    "Otros": "Otro",
    "Propiedad Agrícola/Ganadera": "Propiedad Agrícola/Ganadera",
    "Terreno": "Terreno",
    "Terreno Comercial": "Terreno",
}

CAMPOS_REQUERIDOS = {
    "Casa": ("construction_area_m", "land_m2", "dormitorios", "banos"),
    "Departamento": ("construction_area_m", "dormitorios", "banos"),
    "Terreno": ("total_area",),
}

ETIQUETAS_CAMPOS = {
    "construction_area_m": "m2_construidos nulo o menor/igual a 0",
    "land_m2": "m2_terreno nulo o menor/igual a 0",
    "total_area": "m2_total nulo o menor/igual a 0",
    "dormitorios": "dormitorios nulo o menor/igual a 0",
    "banos": "baños nulo o menor/igual a 0",
}

# Todos los importes del CRM están expresados en dólares estadounidenses (USD).
PRECIO_MINIMO_VENTA_USD = {
    "Casa": 7200,
    "Departamento": 4300,
    "Terreno": 1500,
    "Local Comercial": 2900,
    "Oficina": 2900,
    "Otro": 1500,
    "Propiedad Agrícola/Ganadera": 1500,
}

PRECIO_MINIMO_ALQUILER_USD = {
    "Casa": 72,
    "Departamento": 115,
    "Local Comercial": 72,
    "Oficina": 72,
    "Otro": 72,
    "Propiedad Agrícola/Ganadera": 72,
}

COLUMNAS_REQUERIDAS = {
    "id_propiedad",
    "mlsid",
    "subtipo_original",
    "tipo_transaccion",
    "latitude",
    "longitude",
    "construction_area_m",
    "total_area",
    "land_m2",
    "dormitorios",
    "banos",
    "precio_publicacion",
    "precio_cierre",
}

COLUMNAS_EXCLUSION = [
    "id_propiedad",
    "mlsid",
    "subtipo_original",
    "tipo_propiedad",
    "segmento",
    "tipo_transaccion",
    "motivo",
]


def _exclusiones(
    df: pd.DataFrame,
    mask: pd.Series,
    motivo: str | pd.Series,
) -> pd.DataFrame:
    """Construye un reporte uniforme para las filas marcadas."""

    if not mask.any():
        return pd.DataFrame(columns=COLUMNAS_EXCLUSION)

    resultado = df.loc[mask].reindex(columns=COLUMNAS_EXCLUSION[:-1]).copy()
    if isinstance(motivo, pd.Series):
        resultado["motivo"] = motivo.loc[mask].to_numpy()
    else:
        resultado["motivo"] = motivo
    return resultado.reset_index(drop=True)


def _valor_no_positivo(serie: pd.Series) -> pd.Series:
    numerica = pd.to_numeric(serie, errors="coerce")
    return numerica.isna() | numerica.le(0)


def _filtrar_por_campos_requeridos(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Excluye filas inválidas sin recorrer el DataFrame fila por fila."""

    invalida = pd.Series(False, index=df.index)
    exclusiones: list[pd.DataFrame] = []

    for tipo, campos in CAMPOS_REQUERIDOS.items():
        es_tipo = df["tipo_propiedad"].eq(tipo)
        for campo in campos:
            mask = es_tipo & _valor_no_positivo(df[campo])
            invalida |= mask
            exclusiones.append(_exclusiones(df, mask, ETIQUETAS_CAMPOS[campo]))

    detalle = pd.concat(exclusiones, ignore_index=True)
    return df.loc[~invalida].copy(), detalle


def _asignar_superficies(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza las superficies usadas por cada familia de propiedad."""

    df = df.copy()
    es_casa = df["tipo_propiedad"].eq("Casa")
    es_departamento = df["tipo_propiedad"].eq("Departamento")
    es_terreno = df["tipo_propiedad"].eq("Terreno")

    df["m2_construidos"] = np.select(
        [es_casa | es_departamento, es_terreno],
        [pd.to_numeric(df["construction_area_m"], errors="coerce"), 0],
        default=np.nan,
    )
    terreno_reportado = pd.to_numeric(df["land_m2"], errors="coerce")
    df["m2_terreno"] = np.select(
        [es_casa, es_departamento, es_terreno],
        [
            terreno_reportado,
            terreno_reportado.where(terreno_reportado.gt(0), np.nan),
            pd.to_numeric(df["total_area"], errors="coerce"),
        ],
        default=np.nan,
    )
    return df


def _filtrar_precios_minimos(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    venta = df["tipo_propiedad"].map(PRECIO_MINIMO_VENTA_USD)
    alquiler = df["tipo_propiedad"].map(PRECIO_MINIMO_ALQUILER_USD)
    minimo = venta.where(df["tipo_transaccion"].eq("Venta"), alquiler)
    precio = pd.to_numeric(df["precio_publicacion"], errors="coerce")
    mask = minimo.notna() & (precio.isna() | precio.lt(minimo))
    motivos = pd.Series(
        np.where(
            df["tipo_transaccion"].eq("Venta"),
            "precio menor al mínimo de venta",
            "precio menor al mínimo de alquiler",
        ),
        index=df.index,
    )
    return df.loc[~mask].copy(), _exclusiones(df, mask, motivos)


def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia propiedades y devuelve también el detalle de exclusiones."""

    if df.empty:
        return df.copy(), pd.DataFrame(columns=COLUMNAS_EXCLUSION)

    faltantes = sorted(COLUMNAS_REQUERIDAS.difference(df.columns))
    if faltantes:
        raise ValueError("Faltan columnas requeridas para limpiar: " + ", ".join(faltantes))

    total_inicial = len(df)
    df = df.copy()
    df["tipo_propiedad"] = df["subtipo_original"].map(TIPO_PROPIEDAD_MAP).fillna("Otro")
    reportes: list[pd.DataFrame] = []

    sin_id = df["id_propiedad"].isna() | df["id_propiedad"].astype(str).str.strip().eq("")
    reportes.append(_exclusiones(df, sin_id, "id_propiedad vacío"))
    df = df.loc[~sin_id].copy()

    duplicados = df.duplicated(subset="id_propiedad", keep="last")
    reportes.append(_exclusiones(df, duplicados, "duplicado"))
    df = df.loc[~duplicados].copy()
    log.info("  Duplicados eliminados: %s", int(duplicados.sum()))

    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")
    sin_coordenadas = ~latitude.between(-90, 90) | ~longitude.between(-180, 180)
    reportes.append(_exclusiones(df, sin_coordenadas, "coordenadas ausentes o inválidas"))
    df = df.loc[~sin_coordenadas].copy()
    log.info("  Sin coordenadas válidas descartados: %s", int(sin_coordenadas.sum()))

    df = _asignar_superficies(df)

    antes = len(df)
    df, campos_invalidos = _filtrar_por_campos_requeridos(df)
    reportes.append(campos_invalidos)
    log.info("  Excluidos por campos requeridos: %s", antes - len(df))

    df, outliers_precio = _remover_outliers(df, "precio_publicacion")
    reportes.append(pd.DataFrame(outliers_precio, columns=COLUMNAS_EXCLUSION))
    df, outliers_m2 = _remover_outliers(df, "m2_construidos")
    reportes.append(pd.DataFrame(outliers_m2, columns=COLUMNAS_EXCLUSION))

    antes = len(df)
    df, precios_bajos = _filtrar_precios_minimos(df)
    reportes.append(precios_bajos)
    log.info("  Precios mínimos eliminados: %s", antes - len(df))

    df_excluidos = pd.concat(reportes, ignore_index=True).reindex(columns=COLUMNAS_EXCLUSION)

    for transaccion, cantidad in df["tipo_transaccion"].value_counts().items():
        log.info("  %s: %s registros", transaccion, cantidad)

    log.info(
        "  Total después de limpieza: %s (removidos: %s)",
        len(df),
        total_inicial - len(df),
    )
    if not df_excluidos.empty:
        resumen = (
            df_excluidos.drop_duplicates(subset=["id_propiedad", "motivo"])["motivo"]
            .value_counts()
            .to_string()
        )
        log.info("Resumen exclusiones:\n%s", resumen)

    return df.reset_index(drop=True), df_excluidos


def _remover_outliers(
    df: pd.DataFrame,
    columna: str,
) -> tuple[pd.DataFrame, list[Mapping[str, object]]]:
    """Quita valores fuera de tres desviaciones dentro de grupos comparables."""

    if df.empty:
        return df.copy(), []

    columna_real = "precio_cierre" if columna == "precio_publicacion" else columna
    etiqueta = "outlier precio_cierre" if columna == "precio_publicacion" else f"outlier {columna}"
    valores = pd.to_numeric(df[columna_real], errors="coerce")
    grupos = [df["tipo_propiedad"], df["tipo_transaccion"]]
    conteo = valores.groupby(grupos).transform("count")
    media = valores.groupby(grupos).transform("mean")
    desviacion = valores.groupby(grupos).transform("std")
    fuera_de_rango = valores.lt(media - 3 * desviacion) | valores.gt(media + 3 * desviacion)
    mask = valores.notna() & conteo.ge(4) & desviacion.gt(0) & fuera_de_rango

    detalle = _exclusiones(df, mask, etiqueta)
    limpio = df.loc[~mask].copy().reset_index(drop=True)
    log.info("  Outliers en '%s': %s removidos", columna_real, int(mask.sum()))
    return limpio, detalle.to_dict("records")
