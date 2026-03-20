import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
log = logging.getLogger(__name__)

COLUMNAS_ANALYTICS = [
    "id_propiedad", "mlsid", "tipo_propiedad", "subtipo_original",
    "categoria_propiedad", "estado_propiedad", 
    "segmento",
    "tipo_transaccion",
    "latitude", "longitude", "cluster_zona", "ciudad",
    "m2_construidos", "m2_terreno",
    "dormitorios", "banos", "estacionamientos", "antiguedad",
    "precio_publicacion", "precio_venta", "precio_alquiler_mes", "precio_m2",
    "tiempo_en_mercado", "numero_reducciones",
    "diferencia_vs_promedio_zona", "ratio_activas_vendidas_zona",
    "mes_publicacion", "anio_publicacion", "fecha_venta",
    "status", "transaction_type",
]

# precio_cierre viene del extractor — se mapea según tipo_transaccion
# en _preparar_fila antes de insertar
RENAME_MAP = {
    "land_m2":   "m2_terreno",
    "sold_date": "fecha_venta",
}

INTEGER_MAX = 2147483647
LIMITES = {
    "mes_publicacion":    (1, 12),
    "anio_publicacion":   (1900, 2100),
    "dormitorios":        (0, 50),
    "banos":              (0, 50),
    "estacionamientos":   (0, 200),
    "antiguedad":         (0, 200),
    "tiempo_en_mercado":  (0, INTEGER_MAX),
    "numero_reducciones": (0, INTEGER_MAX),
    "cluster_zona":       (0, 32767),
}

def get_pg_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    )
    return create_engine(url)

def _limpiar_valor(val):
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, pd.Timestamp) and pd.isna(val):
        return None
    if str(val) in ("nan", "NaT", "None", "inf", "-inf"):
        return None
    return val

def _preparar_fila(row: dict) -> dict:
    resultado = {}

    # Separar precio_cierre en precio_venta o precio_alquiler_mes
    # según tipo_transaccion antes de procesar columnas
    tipo_t      = row.get("tipo_transaccion", "Venta")
    precio_cierre = _limpiar_valor(row.get("precio_cierre"))

    if tipo_t == "Venta":
        row["precio_venta"]        = precio_cierre
        row["precio_alquiler_mes"] = None
    else:
        row["precio_venta"]        = None
        row["precio_alquiler_mes"] = precio_cierre

    for col in COLUMNAS_ANALYTICS:
        val = _limpiar_valor(row.get(col))

        # Fechas
        if col == "fecha_venta":
            if val is not None and hasattr(val, "date"):
                val = val.date()
            resultado[col] = val
            continue

        # Enteros con límites
        if col in LIMITES:
            if val is not None:
                try:
                    val = int(float(val))
                    mn, mx = LIMITES[col]
                    if val < mn or val > mx:
                        val = None
                except (ValueError, OverflowError):
                    val = None
            resultado[col] = val
            continue

        resultado[col] = val

    return resultado

def cargar_datos(df: pd.DataFrame):
    engine = get_pg_engine()
    df     = df.rename(columns=RENAME_MAP)

    insertados   = 0
    actualizados = 0
    errores      = 0

    upsert_sql = text("""
        INSERT INTO property_analytics (
            id_propiedad, mlsid, tipo_propiedad, subtipo_original,
            categoria_propiedad, estado_propiedad, segmento,
            tipo_transaccion,
            latitude, longitude, cluster_zona, ciudad,
            m2_construidos, m2_terreno,
            dormitorios, banos, estacionamientos, antiguedad,
            precio_publicacion, precio_venta, precio_alquiler_mes, precio_m2,
            tiempo_en_mercado, numero_reducciones,
            diferencia_vs_promedio_zona, ratio_activas_vendidas_zona,
            mes_publicacion, anio_publicacion, fecha_venta,
            status, transaction_type
        ) VALUES (
            :id_propiedad, :mlsid, :tipo_propiedad, :subtipo_original,
            :categoria_propiedad, :estado_propiedad, :segmento,
            :tipo_transaccion,
            :latitude, :longitude, :cluster_zona, :ciudad,
            :m2_construidos, :m2_terreno,
            :dormitorios, :banos, :estacionamientos, :antiguedad,
            :precio_publicacion, :precio_venta, :precio_alquiler_mes, :precio_m2,
            :tiempo_en_mercado, :numero_reducciones,
            :diferencia_vs_promedio_zona, :ratio_activas_vendidas_zona,
            :mes_publicacion, :anio_publicacion, :fecha_venta,
            :status, :transaction_type
        )
        ON CONFLICT (id_propiedad) DO UPDATE SET
            precio_publicacion          = EXCLUDED.precio_publicacion,
            precio_venta                = EXCLUDED.precio_venta,
            precio_alquiler_mes         = EXCLUDED.precio_alquiler_mes,
            precio_m2                   = EXCLUDED.precio_m2,
            tiempo_en_mercado           = EXCLUDED.tiempo_en_mercado,
            cluster_zona                = EXCLUDED.cluster_zona,
            tipo_transaccion            = EXCLUDED.tipo_transaccion,
            diferencia_vs_promedio_zona = EXCLUDED.diferencia_vs_promedio_zona,
            ratio_activas_vendidas_zona = EXCLUDED.ratio_activas_vendidas_zona,
            numero_reducciones          = EXCLUDED.numero_reducciones,
            status                      = EXCLUDED.status,
            fecha_actualizacion         = NOW()
    """)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                fila   = _preparar_fila(row.to_dict())
                result = conn.execute(upsert_sql, fila)
                if result.rowcount == 1:
                    insertados += 1
                else:
                    actualizados += 1
            except Exception as e:
                errores += 1
                log.warning(f"  Error en id_propiedad={row.get('id_propiedad')}: {e}")
                continue

    if errores > 0:
        log.warning(f"  {errores} filas con error ignoradas")

    return insertados, actualizados

# ------------------------------------------------------------
# Dataset entrenable — separado por tipo de transacción
# ------------------------------------------------------------
def generar_dataset_entrenable(df: pd.DataFrame) -> int:
    os.makedirs("data", exist_ok=True)

    # Renombrar columnas CRM → analíticas
    df = df.rename(columns={
        "land_m2":   "m2_terreno",
        "sold_date": "fecha_venta",
    })

    # Separar precio_cierre en precio_venta / precio_alquiler_mes
    df["precio_venta"] = np.where(
        df["tipo_transaccion"] == "Venta",
        df["precio_cierre"],
        None
    )
    df["precio_alquiler_mes"] = np.where(
        df["tipo_transaccion"] == "Alquiler",
        df["precio_cierre"],
        None
    )

    total = 0

    # Dataset de ventas
    df_ventas = df[
        (df["tipo_transaccion"] == "Venta") &
        (df["precio_venta"].notna())
    ].copy()

    if not df_ventas.empty:
        ruta_ventas = "data/dataset_ventas.parquet"
        df_ventas.to_parquet(ruta_ventas, index=False)
        log.info(f"  Dataset ventas exportado: {ruta_ventas} ({len(df_ventas)} registros)")
        total += len(df_ventas)
    else:
        log.warning("  No hay ventas para exportar")

    # Dataset de alquileres
    df_alquileres = df[
        (df["tipo_transaccion"] == "Alquiler") &
        (df["precio_alquiler_mes"].notna())
    ].copy()

    if not df_alquileres.empty:
        ruta_alquileres = "data/dataset_alquileres.parquet"
        df_alquileres.to_parquet(ruta_alquileres, index=False)
        log.info(f"  Dataset alquileres exportado: {ruta_alquileres} ({len(df_alquileres)} registros)")
        total += len(df_alquileres)
    else:
        log.warning("  No hay alquileres para exportar")

    return total