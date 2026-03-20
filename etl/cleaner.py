import logging
import pandas as pd
import numpy as np
from datetime import date

log = logging.getLogger(__name__)


# Mapeo de subtipos → grupos
TIPO_PROPIEDAD_MAP = {
    "Casa":                             "Casa",
    "Casa de Calidad":                  "Casa",
    "Casa de Campo":                    "Casa",
    "Casa con Espacio Comercial":       "Casa",
    "Departamento":                     "Departamento",
    "Dúplex":                           "Departamento",
    "Penthouse":                        "Departamento",
    "Estudio/Monoambiente":             "Departamento",
    "Condominio / Departamento":        "Departamento",
    "Apartamento con servicio de hotel":"Departamento",
    "Local Comercial":                  "Local Comercial",
    "Comercial/Negocio":                "Local Comercial",
    "Oficina":                          "Oficina",
    "Baulera":                          "Otro",
    "Clínica de salud":                 "Otro",
    "Edificio":                         "Otro",
    "Edificio de apartamentos entero":  "Otro",
    "Edificio/Construcción":            "Otro",
    "Galpon":                           "Otro",
    "Garaje/Baulera":                   "Otro",
    "Hotel/Edificio de apartamentos":   "Otro",
    "Quinta":                           "Otro",
    "Otros":                            "Otro",
    "Propiedad Agrícola/Ganadera":      "Propiedad Agrícola/Ganadera",
    "Terreno":                          "Terreno",
    "Terreno Comercial":                "Terreno",
}

# Campos requeridos por tipo
CAMPOS_REQUERIDOS = {
    "Casa":         ["construction_area_m", "land_m2", "dormitorios", "banos"],
    "Departamento": ["construction_area_m", "dormitorios", "banos"],
    "Terreno":      ["total_area"],
}

# Precio mínimo en BOB por tipo — VENTA
PRECIO_MINIMO_VENTA = {
    "Casa":                        50000,
    "Departamento":                30000,
    "Terreno":                     10000,
    "Local Comercial":             20000,
    "Oficina":                     20000,
    "Otro":                        10000,
    "Propiedad Agrícola/Ganadera": 10000,
}

# Precio mínimo en BOB/mes por tipo — ALQUILER
PRECIO_MINIMO_ALQUILER = {
    "Casa":                         500,
    "Departamento":                 800,
    "Local Comercial":              500,
    "Oficina":                      500,
    "Otro":                         500,
    "Propiedad Agrícola/Ganadera":  500,
}


def _generar_motivos_exclusion(row: pd.Series) -> list:
    tipo   = row.get("tipo_propiedad", "Otro")
    campos = CAMPOS_REQUERIDOS.get(tipo, [])

    LABELS = {
        "construction_area_m": "m2_construidos nulo o 0",
        "land_m2":             "m2_terreno libre nulo o 0",
        "total_area":          "m2_total nulo o 0",
        "dormitorios":         "dormitorios nulo o 0",
        "banos":               "baños nulo o 0",
    }

    return [
        LABELS.get(campo, f"{campo} inválido")
        for campo in campos
        if (lambda v: v is None or (isinstance(v, float) and np.isnan(v)) or v == 0)(row.get(campo))
    ]

def _filtrar_por_campos_requeridos(df: pd.DataFrame):
    excluidos = []
    validos   = []

    for _, row in df.iterrows():
        motivos = _generar_motivos_exclusion(row)
        if motivos:
                    for motivo in motivos:
                        excluidos.append({
                            "id_propiedad":     row.get("id_propiedad"),
                            "mlsid":            row.get("mlsid"),
                            "subtipo_original": row.get("subtipo_original"),
                            "tipo_propiedad":   row.get("tipo_propiedad"),
                            "tipo_transaccion": row.get("tipo_transaccion"),
                            "motivo":           motivo,
                        })
        else:
            validos.append(row)

    df_valido    = pd.DataFrame(validos).reset_index(drop=True) if validos else df.iloc[0:0]
    df_excluidos = pd.DataFrame(excluidos)
    return df_valido, df_excluidos

def limpiar_datos(df: pd.DataFrame):
    total_inicial = len(df)
    todos_excluidos = []

    # 1 — Duplicados
    mask_dup = df.duplicated(subset="id_propiedad", keep="last")
    df_dup   = df[mask_dup].copy()
    df       = df[~mask_dup]
    log.info(f"  Duplicados eliminados: {mask_dup.sum()}")

    # 2 — Sin coordenadas
    sin_coords = df["latitude"].isna() | df["longitude"].isna()
    df = df[~sin_coords]
    log.info(f"  Sin coordenadas descartados: {sin_coords.sum()}")
    
    # 3 — Tipo de propiedad
    df["tipo_propiedad"] = df["subtipo_original"].map(TIPO_PROPIEDAD_MAP).fillna("Otro")
    df_dup["tipo_propiedad"] = df_dup["subtipo_original"].map(TIPO_PROPIEDAD_MAP).fillna("Otro")
    
        # Registrar duplicados ahora que tipo_propiedad existe
    for _, row in df_dup.iterrows():
        todos_excluidos.append({
            "id_propiedad":     row.get("id_propiedad"),
            "mlsid":            row.get("mlsid"),
            "subtipo_original": row.get("subtipo_original", ""),
            "tipo_propiedad":   row.get("tipo_propiedad", ""),
            "segmento":         row.get("segmento", ""),
            "tipo_transaccion": row.get("tipo_transaccion", ""),
            "motivo":           "duplicado",
        })


    # 4 — Calcular m2_construidos según tipo
    # Casa        → construction_area_m (área ocupada)
    # Departamento → construction_area_m (área ocupada)
    # Terreno/resto → total_area (superficie total)
    es_depto_o_casa = df["subtipo_original"].isin([
        "Departamento", "Dúplex", "Penthouse",
        "Estudio/Monoambiente", "Condominio / Departamento",
        "Apartamento con servicio de hotel",
        "Casa", "Casa de Calidad", "Casa de Campo",
        "Casa con Espacio Comercial",
    ])
    df["m2_construidos"] = np.where(
        es_depto_o_casa,
        df["construction_area_m"],
        df["total_area"]
    )
    mask_sin_m2 = (df["m2_construidos"] <= 0) & \
                  (~df["tipo_propiedad"].isin(["Casa", "Departamento", "Terreno"]))
    for _, row in df[mask_sin_m2].iterrows():
        todos_excluidos.append({
            "id_propiedad":     row.get("id_propiedad"),
            "mlsid":            row.get("mlsid"),
            "subtipo_original": row.get("subtipo_original"),
            "tipo_propiedad":   row.get("tipo_propiedad", ""),
            "tipo_transaccion": row.get("tipo_transaccion", ""),
            "motivo":           "m2_construidos <= 0 o nulo",
        })
    df = df[~mask_sin_m2]
    log.info(f"  Sin m2 válidos descartados: {mask_sin_m2.sum()}")



    # 5 — Filtro por campos requeridos (una fila por motivo)
    antes_filtro = len(df)
    df, excluidos_campos = _filtrar_por_campos_requeridos(df)
    todos_excluidos.extend(excluidos_campos.to_dict("records") if not excluidos_campos.empty else [])
    log.info(f"  Excluidos por campos requeridos: {antes_filtro - len(df)}")

    # 6 — Outliers
    df, excluidos_outliers = _remover_outliers(df, "precio_publicacion")
    todos_excluidos.extend(excluidos_outliers)
    df, excluidos_outliers_m2 = _remover_outliers(df, "m2_construidos")
    todos_excluidos.extend(excluidos_outliers_m2)

    # 7 — Precio mínimo
    antes = len(df)
    df_venta    = df[df["tipo_transaccion"] == "Venta"].copy()
    df_alquiler = df[df["tipo_transaccion"] == "Alquiler"].copy()

    for tipo, precio_min in PRECIO_MINIMO_VENTA.items():
        mask = (df_venta["tipo_propiedad"] == tipo) & (df_venta["precio_publicacion"] < precio_min)
        for _, row in df_venta[mask].iterrows():
            todos_excluidos.append({
                "id_propiedad":     row.get("id_propiedad"),
                "mlsid":            row.get("mlsid"),
                "subtipo_original": row.get("subtipo_original"),
                "tipo_propiedad":   row.get("tipo_propiedad"),
                "tipo_transaccion": row.get("tipo_transaccion"),
                "motivo":           "precio menor mínimo venta",
            })
        df_venta = df_venta[~mask]

    for tipo, precio_min in PRECIO_MINIMO_ALQUILER.items():
        mask = (df_alquiler["tipo_propiedad"] == tipo) & (df_alquiler["precio_publicacion"] < precio_min)
        for _, row in df_alquiler[mask].iterrows():
            todos_excluidos.append({
                "id_propiedad":     row.get("id_propiedad"),
                "mlsid":            row.get("mlsid"),
                "subtipo_original": row.get("subtipo_original"),
                "tipo_propiedad":   row.get("tipo_propiedad"),
                "tipo_transaccion": row.get("tipo_transaccion"),
                "motivo":           "precio menor mínimo alquiler",
            })
        df_alquiler = df_alquiler[~mask]

    df = pd.concat([df_venta, df_alquiler], ignore_index=True)
    log.info(f"  Precios mínimos eliminados: {antes - len(df)}")

    # Resumen
    df_excluidos = pd.DataFrame(todos_excluidos)

    for tipo_t in df["tipo_transaccion"].unique():
        n = len(df[df["tipo_transaccion"] == tipo_t])
        log.info(f"  {tipo_t}: {n} registros")

    log.info(f"  Total después de limpieza: {len(df)} (removidos: {total_inicial - len(df)})")

    if not df_excluidos.empty:
        log.info("Resumen exclusiones:")
        resumen_log = (
            df_excluidos
            .drop_duplicates(subset=["id_propiedad", "motivo"])
            ["motivo"]
            .value_counts()
        )
        log.info(resumen_log.to_string())

    return df.reset_index(drop=True), df_excluidos

# def _remover_outliers(df, columna):
#     resultado = []
#     excluidos = []

#     for (tipo, transaccion), grupo in df.groupby(["tipo_propiedad", "tipo_transaccion"]):
#         media = grupo[columna].mean()
#         std   = grupo[columna].std()

#         # ✅ Si std es NaN (grupo de 1 elemento), conservar todo el grupo
#         if pd.isna(std) or std == 0:
#             resultado.append(grupo)
#             continue

#         mask = (
#             (grupo[columna] >= media - 3 * std) &
#             (grupo[columna] <= media + 3 * std)
#         )

#         resultado.append(grupo[mask])

#         excl = grupo[~mask].copy()
#         excl["motivo"] = f"outlier {columna}"
#         excluidos.append(excl)

#     df_limpio    = pd.concat(resultado).reset_index(drop=True)
#     df_excluidos = pd.concat(excluidos).reset_index(drop=True) if excluidos else pd.DataFrame()

#     return df_limpio, df_excluidos

def _remover_outliers(df: pd.DataFrame, columna: str):
    antes     = len(df)
    resultado = []
    excluidos = []

    # Para precio usamos precio_cierre (real de transacción), no precio_publicacion
    col_real = "precio_cierre" if columna == "precio_publicacion" else columna
    label    = "outlier precio_cierre" if columna == "precio_publicacion" else f"outlier {columna}"

    for (tipo, transaccion), grupo in df.groupby(["tipo_propiedad", "tipo_transaccion"]):
        serie = grupo[col_real].dropna()

        if len(serie) < 4 or pd.isna(serie.std()) or serie.std() == 0:
            resultado.append(grupo)
            continue

        media = serie.mean()
        std   = serie.std()

        mask_ok = (
            grupo[col_real].isna() |
            (
                (grupo[col_real] >= media - 3 * std) &
                (grupo[col_real] <= media + 3 * std)
            )
        )
        resultado.append(grupo[mask_ok])

        for _, row in grupo[~mask_ok].iterrows():
            excluidos.append({
                "id_propiedad":     row.get("id_propiedad"),
                "mlsid":            row.get("mlsid"),
                "subtipo_original": row.get("subtipo_original"),
                "tipo_propiedad":   row.get("tipo_propiedad"),
                "tipo_transaccion": row.get("tipo_transaccion"),
                "motivo":           label,
            })

    df_limpio = pd.concat(resultado).reset_index(drop=True)
    log.info(f"  Outliers en '{col_real}': {antes - len(df_limpio)} removidos")
    return df_limpio, excluidos