"""Carga transaccional de datos limpios en PostgreSQL."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator, Sequence
from numbers import Real

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, bindparam, text
from sqlalchemy.exc import SQLAlchemyError

from .config import get_pg_engine

log = logging.getLogger(__name__)

COLUMNAS_ANALYTICS = [
    "id_propiedad",
    "mlsid",
    "tipo_propiedad",
    "subtipo_original",
    "estado_propiedad",
    "segmento",
    "tipo_transaccion",
    "latitude",
    "longitude",
    "cluster_zona",
    "ciudad",
    "m2_construidos",
    "m2_terreno",
    "dormitorios",
    "banos",
    "estacionamientos",
    "antiguedad",
    "precio_publicacion",
    "precio_venta",
    "precio_alquiler_mes",
    "precio_m2",
    "tiempo_en_mercado",
    "numero_reducciones",
    "diferencia_vs_promedio_zona",
    "ratio_activas_vendidas_zona",
    "mes_publicacion",
    "anio_publicacion",
    "fecha_venta",
    "status",
]

# m2_terreno ya se calcula en cleaner.py; renombrar land_m2 encima de esa
# columna perdía el área correcta de los terrenos.
RENAME_MAP = {"sold_date": "fecha_venta"}

INTEGER_MAX = 2_147_483_647
LIMITES = {
    "mes_publicacion": (1, 12),
    "anio_publicacion": (1900, 2100),
    "dormitorios": (0, 50),
    "banos": (0, 50),
    "estacionamientos": (0, 200),
    "antiguedad": (0, 200),
    "tiempo_en_mercado": (0, INTEGER_MAX),
    "numero_reducciones": (0, INTEGER_MAX),
    "cluster_zona": (0, 32_767),
}

UPSERT_SQL = text("""
    INSERT INTO property_analytics (
        id_propiedad, mlsid, tipo_propiedad, subtipo_original,
        estado_propiedad, segmento, tipo_transaccion,
        latitude, longitude, cluster_zona, ciudad,
        m2_construidos, m2_terreno,
        dormitorios, banos, estacionamientos, antiguedad,
        precio_publicacion, precio_venta, precio_alquiler_mes, precio_m2,
        tiempo_en_mercado, numero_reducciones,
        diferencia_vs_promedio_zona, ratio_activas_vendidas_zona,
        mes_publicacion, anio_publicacion, fecha_venta, status
    ) VALUES (
        :id_propiedad, :mlsid, :tipo_propiedad, :subtipo_original,
        :estado_propiedad, :segmento, :tipo_transaccion,
        :latitude, :longitude, :cluster_zona, :ciudad,
        :m2_construidos, :m2_terreno,
        :dormitorios, :banos, :estacionamientos, :antiguedad,
        :precio_publicacion, :precio_venta, :precio_alquiler_mes, :precio_m2,
        :tiempo_en_mercado, :numero_reducciones,
        :diferencia_vs_promedio_zona, :ratio_activas_vendidas_zona,
        :mes_publicacion, :anio_publicacion, :fecha_venta, :status
    )
    ON CONFLICT (id_propiedad) DO UPDATE SET
        mlsid                        = EXCLUDED.mlsid,
        tipo_propiedad               = EXCLUDED.tipo_propiedad,
        subtipo_original             = EXCLUDED.subtipo_original,
        estado_propiedad             = EXCLUDED.estado_propiedad,
        segmento                     = EXCLUDED.segmento,
        latitude                     = EXCLUDED.latitude,
        longitude                    = EXCLUDED.longitude,
        ciudad                       = EXCLUDED.ciudad,
        m2_construidos               = EXCLUDED.m2_construidos,
        m2_terreno                   = EXCLUDED.m2_terreno,
        dormitorios                  = EXCLUDED.dormitorios,
        banos                        = EXCLUDED.banos,
        estacionamientos             = EXCLUDED.estacionamientos,
        antiguedad                   = EXCLUDED.antiguedad,
        precio_publicacion           = EXCLUDED.precio_publicacion,
        precio_venta                 = EXCLUDED.precio_venta,
        precio_alquiler_mes          = EXCLUDED.precio_alquiler_mes,
        precio_m2                    = EXCLUDED.precio_m2,
        tiempo_en_mercado            = EXCLUDED.tiempo_en_mercado,
        cluster_zona                 = EXCLUDED.cluster_zona,
        tipo_transaccion             = EXCLUDED.tipo_transaccion,
        diferencia_vs_promedio_zona  = EXCLUDED.diferencia_vs_promedio_zona,
        ratio_activas_vendidas_zona  = EXCLUDED.ratio_activas_vendidas_zona,
        numero_reducciones           = EXCLUDED.numero_reducciones,
        mes_publicacion              = EXCLUDED.mes_publicacion,
        anio_publicacion             = EXCLUDED.anio_publicacion,
        fecha_venta                  = EXCLUDED.fecha_venta,
        status                       = EXCLUDED.status,
        fecha_actualizacion          = NOW()
""")

IDS_EXISTENTES_SQL = text("""
    SELECT id_propiedad
    FROM property_analytics
    WHERE id_propiedad IN :ids
""").bindparams(bindparam("ids", expanding=True))


def _limpiar_valor(valor: object) -> object:
    """Convierte valores de pandas/numpy a escalares aceptados por psycopg2."""

    if valor is None:
        return None
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, Real) and not math.isfinite(float(valor)):
        return None
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(valor, str) and valor.strip().lower() in {
        "nan",
        "nat",
        "none",
        "inf",
        "-inf",
    }:
        return None
    return valor


def _preparar_fila(row: dict[str, object]) -> dict[str, object]:
    """Adapta una fila transformada al esquema de property_analytics."""

    tipo_transaccion = row.get("tipo_transaccion", "Venta")
    precio_cierre = _limpiar_valor(row.get("precio_cierre"))
    precios = {
        "precio_venta": precio_cierre if tipo_transaccion == "Venta" else None,
        "precio_alquiler_mes": (precio_cierre if tipo_transaccion == "Alquiler" else None),
    }

    resultado: dict[str, object] = {}
    for columna in COLUMNAS_ANALYTICS:
        valor = precios.get(columna, _limpiar_valor(row.get(columna)))

        if columna == "fecha_venta" and valor is not None and hasattr(valor, "date"):
            valor = valor.date()  # type: ignore[union-attr]

        if columna in LIMITES and valor is not None:
            try:
                valor = int(float(valor))
                minimo, maximo = LIMITES[columna]
                if not minimo <= valor <= maximo:
                    valor = None
            except (TypeError, ValueError, OverflowError):
                valor = None

        resultado[columna] = valor

    return resultado


def _lotes(
    filas: Sequence[dict[str, object]],
    tamano: int,
) -> Iterator[list[dict[str, object]]]:
    for inicio in range(0, len(filas), tamano):
        yield list(filas[inicio : inicio + tamano])


def _ejecutar_lote(
    connection: Connection,
    filas: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Ejecuta en bloque y, ante un dato malo, aísla solo esa fila."""

    try:
        with connection.begin_nested():
            connection.execute(UPSERT_SQL, filas)
        return filas
    except SQLAlchemyError as exc:
        log.warning(
            "  Falló un lote de %s filas; se validarán individualmente: %s",
            len(filas),
            exc,
        )

    exitosas: list[dict[str, object]] = []
    for fila in filas:
        try:
            with connection.begin_nested():
                connection.execute(UPSERT_SQL, fila)
            exitosas.append(fila)
        except SQLAlchemyError as exc:
            log.warning(
                "  Error en id_propiedad=%s: %s",
                fila.get("id_propiedad"),
                exc,
            )
    return exitosas


def cargar_datos(
    df: pd.DataFrame,
    engine: Engine | None = None,
    batch_size: int = 1_000,
) -> tuple[int, int]:
    """Hace upsert por lotes y devuelve conteos reales de insertados/actualizados."""

    if df.empty:
        return 0, 0
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero")

    engine = engine or get_pg_engine()
    preparado = df.rename(columns=RENAME_MAP).drop_duplicates(subset="id_propiedad", keep="last")
    filas = [_preparar_fila(row) for row in preparado.to_dict("records")]

    insertados = 0
    actualizados = 0
    errores = 0

    with engine.begin() as connection:
        for lote in _lotes(filas, batch_size):
            ids = [fila["id_propiedad"] for fila in lote if fila["id_propiedad"] is not None]
            existentes = (
                {row[0] for row in connection.execute(IDS_EXISTENTES_SQL, {"ids": ids})}
                if ids
                else set()
            )

            exitosas = _ejecutar_lote(connection, lote)
            errores += len(lote) - len(exitosas)
            insertados += sum(fila["id_propiedad"] not in existentes for fila in exitosas)
            actualizados += sum(fila["id_propiedad"] in existentes for fila in exitosas)

    if errores:
        log.warning("  %s filas con error fueron ignoradas", errores)
    return insertados, actualizados


def generar_dataset_entrenable(df: pd.DataFrame) -> int:
    """Exporta datasets separados para modelos de venta y alquiler."""

    os.makedirs("data", exist_ok=True)
    df = df.rename(columns=RENAME_MAP).copy()
    df["precio_venta"] = df["precio_cierre"].where(df["tipo_transaccion"].eq("Venta"))
    df["precio_alquiler_mes"] = df["precio_cierre"].where(df["tipo_transaccion"].eq("Alquiler"))

    total = 0
    configuraciones = (
        ("Venta", "precio_venta", "data/dataset_ventas.parquet"),
        ("Alquiler", "precio_alquiler_mes", "data/dataset_alquileres.parquet"),
    )
    for transaccion, columna_precio, ruta in configuraciones:
        subset = df.loc[df["tipo_transaccion"].eq(transaccion) & df[columna_precio].notna()].copy()
        if subset.empty:
            log.warning("  No hay registros de %s para exportar", transaccion.lower())
            continue
        subset.to_parquet(ruta, index=False)
        log.info("  Dataset exportado: %s (%s registros)", ruta, len(subset))
        total += len(subset)

    return total
