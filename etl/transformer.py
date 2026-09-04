import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from .config import get_crm_engine, get_pg_engine

log = logging.getLogger(__name__)


# ------------------------------------------------------------
# Transformaciones principales
# ------------------------------------------------------------
def transformar_datos(
    df: pd.DataFrame,
    pg_engine: Engine | None = None,
    crm_engine: Engine | None = None,
) -> pd.DataFrame:
    """Calcula las variables analíticas sin modificar el DataFrame recibido."""

    df = df.copy()
    hoy = date.today()
    pg_engine = pg_engine or get_pg_engine()
    centroides = _cargar_centroides(pg_engine)

    # 1 — Antigüedad
    df["year_construction"] = pd.to_datetime(df["year_construction"], errors="coerce").dt.year
    df["antiguedad"] = hoy.year - df["year_construction"]
    df["antiguedad"] = df["antiguedad"].clip(lower=0)

    # 2 — Tiempo en mercado
    df["date_of_listing"] = pd.to_datetime(df["date_of_listing"], errors="coerce")
    df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce")
    df["cancellation_date"] = pd.to_datetime(df["cancellation_date"], errors="coerce")
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"], errors="coerce")

    df["tiempo_en_mercado"] = np.where(
        df["sold_date"].notna(),
        (df["sold_date"] - df["date_of_listing"]).dt.days,
        np.where(
            df["cancellation_date"].notna(),
            (df["cancellation_date"] - df["date_of_listing"]).dt.days,
            np.where(
                df["contract_end_date"].notna(),
                (df["contract_end_date"] - df["date_of_listing"]).dt.days,
                (pd.Timestamp(hoy) - df["date_of_listing"]).dt.days,
            ),
        ),
    )
    df["tiempo_en_mercado"] = df["tiempo_en_mercado"].clip(lower=0)

    # 3 — Mes y año de publicación
    df["mes_publicacion"] = df["date_of_listing"].dt.month
    df["anio_publicacion"] = df["date_of_listing"].dt.year

    # 4 — Asignar cluster_zona por ciudad
    df = _asignar_clusters_multiciudad(df, centroides=centroides)

    # 5 — precio_m2, vectorizado para evitar un callback Python por fila.
    df["precio_m2"] = calcular_precio_m2(df)

    # 6 — Features de zona por ciudad
    df = _calcular_features_zona_multiciudad(
        df,
        crm_engine=crm_engine,
        pg_engine=pg_engine,
        centroides=centroides,
    )

    # 7 — numero_reducciones (default 0 hasta tener historial)
    df["numero_reducciones"] = 0

    return df


def _cargar_centroides(pg_engine: Engine) -> pd.DataFrame:
    """Carga una vez los centroides usados por todas las transformaciones."""

    with pg_engine.connect() as conn:
        return pd.read_sql(
            "SELECT cluster_id, ciudad, centroide_lat, centroide_lng FROM zona_clusters",
            conn,
        )


def calcular_precio_m2(df: pd.DataFrame) -> pd.Series:
    """Calcula precio por m² usando cierre y, si falta, publicación."""

    subtipos_departamento = {
        "departamento",
        "dúplex",
        "penthouse",
        "estudio/monoambiente",
        "condominio / departamento",
        "apartamento con servicio de hotel",
    }
    es_departamento = (
        df["subtipo_original"].fillna("").astype(str).str.lower().isin(subtipos_departamento)
    )
    construidos = pd.to_numeric(df["construction_area_m"], errors="coerce")
    totales = pd.to_numeric(df["total_area"], errors="coerce")
    superficie = construidos.where(es_departamento, totales)

    cierre = pd.to_numeric(df["precio_cierre"], errors="coerce")
    publicacion = pd.to_numeric(df["precio_publicacion"], errors="coerce")
    precio = cierre.where(cierre.gt(0), publicacion)

    valido = superficie.gt(0) & precio.gt(0)
    return (precio / superficie).where(valido)


# ------------------------------------------------------------
# Asignar cluster_zona respetando ciudad
# Cada ciudad tiene sus propios centroides
# ------------------------------------------------------------
def _asignar_clusters_multiciudad(
    df: pd.DataFrame,
    pg_engine: Engine | None = None,
    centroides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from sklearn.metrics import pairwise_distances

    if centroides is None:
        centroides = _cargar_centroides(pg_engine or get_pg_engine())

    if centroides.empty:
        log.warning(
            "  zona_clusters está vacía — cluster_zona será NULL. "
            "Ejecutá: python -m etl.clustering.setup_clusters"
        )
        df["cluster_zona"] = None
        return df

    df["cluster_zona"] = None
    ciudades_sin_clusters = []

    for ciudad in df["ciudad"].unique():
        cents_ciudad = centroides[centroides["ciudad"] == ciudad]

        if len(cents_ciudad) == 0:
            ciudades_sin_clusters.append(ciudad)
            continue

        mask = df["ciudad"] == ciudad
        coords = df.loc[mask, ["latitude", "longitude"]].values

        if len(coords) == 0:
            continue

        matriz_centroides = cents_ciudad[["centroide_lat", "centroide_lng"]].values
        distancias = pairwise_distances(coords, matriz_centroides, metric="euclidean")
        indices = distancias.argmin(axis=1)

        df.loc[mask, "cluster_zona"] = cents_ciudad["cluster_id"].values[indices]

    if ciudades_sin_clusters:
        log.warning(f"  Sin clusters para: {', '.join(ciudades_sin_clusters)}")

    asignados = df["cluster_zona"].notna().sum()
    log.info(
        "  Clusters asignados: %s/%s propiedades en %s ciudades",
        asignados,
        len(df),
        df["ciudad"].nunique(),
    )
    return df


# ------------------------------------------------------------
# Features de zona por ciudad
# ratio_activas_vendidas_zona y diferencia_vs_promedio_zona
# ------------------------------------------------------------
def _calcular_features_zona_multiciudad(
    df: pd.DataFrame,
    crm_engine: Engine | None = None,
    pg_engine: Engine | None = None,
    centroides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from sklearn.metrics import pairwise_distances

    crm_engine = crm_engine or get_crm_engine()
    if centroides is None:
        centroides = _cargar_centroides(pg_engine or get_pg_engine())

    def asignar_cluster_a(df_coords: pd.DataFrame, ciudad: str) -> pd.DataFrame:
        cents = centroides[centroides["ciudad"] == ciudad]
        if len(cents) == 0:
            df_coords["cluster_zona"] = None
            return df_coords
        coords = df_coords[["latitude", "longitude"]].values
        distancias = pairwise_distances(
            coords, cents[["centroide_lat", "centroide_lng"]].values, metric="euclidean"
        )
        indices = distancias.argmin(axis=1)
        df_coords = df_coords.copy()
        df_coords["cluster_zona"] = cents["cluster_id"].values[indices]
        return df_coords

    # Inicializar columnas
    df["ratio_activas_vendidas_zona"] = 0.0
    df["diferencia_vs_promedio_zona"] = 0.0

    ciudades = df["ciudad"].unique().tolist()

    for ciudad in ciudades:
        log.info(f"  Calculando features de zona: {ciudad}...")

        cents_ciudad = centroides[centroides["ciudad"] == ciudad]
        if len(cents_ciudad) == 0:
            log.warning(f"  Sin centroides para {ciudad} — features de zona en 0")
            continue

        lat_min = cents_ciudad["centroide_lat"].min() - 0.2
        lat_max = cents_ciudad["centroide_lat"].max() + 0.2
        lng_min = cents_ciudad["centroide_lng"].min() - 0.2
        lng_max = cents_ciudad["centroide_lng"].max() + 0.2

        params = {
            "ciudad": ciudad,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lng_min": lng_min,
            "lng_max": lng_max,
        }

        # ================================================================
        # VENTA — ratio activas/vendidas y promedio precio_m2
        # ================================================================
        ratio_venta_query = text("""
            SELECT loc.latitude, loc.longitude, sl.name AS status
            FROM listings l
            JOIN listing_transaction_types ltt ON l.transaction_type_id = ltt.id
            JOIN status_listings sl            ON l.status_listing_id = sl.id
            JOIN locations loc                 ON loc.listing_id = l.id
            JOIN cities ci                     ON loc.city_id = ci.id
            WHERE ltt.id = 1
              AND ci.name = :ciudad
              AND loc.latitude  BETWEEN :lat_min AND :lat_max
              AND loc.longitude BETWEEN :lng_min AND :lng_max
              AND sl.name IN ('Activa', 'Venta Aceptada/Vendida')
        """)

        promedio_venta_query = text("""
            SELECT
                loc.latitude,
                loc.longitude,
                (lp.amount / NULLIF(li.construction_area_m, 0)) AS precio_m2
            FROM listings l
            JOIN listing_transaction_types ltt ON l.transaction_type_id = ltt.id
            JOIN listings_information li       ON li.listing_id = l.id
            JOIN listing_prices lp             ON lp.listing_id = l.id
            JOIN locations loc                 ON loc.listing_id = l.id
            JOIN cities ci                     ON loc.city_id = ci.id
            JOIN transactions t ON t.listing_id = l.id
                AND t.transaction_type_id = 1
                AND t.transaction_status_id IN (2, 5)
            WHERE ltt.id = 1
              AND ci.name = :ciudad
              AND loc.latitude  BETWEEN :lat_min AND :lat_max
              AND loc.longitude BETWEEN :lng_min AND :lng_max
              AND t.sold_date >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
              AND li.construction_area_m > 0
              AND lp.amount > 0
        """)

        # ================================================================
        # ALQUILER — ratio activas/alquiladas y promedio precio_mes/m2
        # ================================================================
        ratio_alquiler_query = text("""
            SELECT loc.latitude, loc.longitude, sl.name AS status
            FROM listings l
            JOIN listing_transaction_types ltt ON l.transaction_type_id = ltt.id
            JOIN status_listings sl            ON l.status_listing_id = sl.id
            JOIN locations loc                 ON loc.listing_id = l.id
            JOIN cities ci                     ON loc.city_id = ci.id
            WHERE ltt.id = 2
              AND ci.name = :ciudad
              AND loc.latitude  BETWEEN :lat_min AND :lat_max
              AND loc.longitude BETWEEN :lng_min AND :lng_max
              AND sl.name IN ('Activa', 'Alquilado')
        """)

        promedio_alquiler_query = text("""
            SELECT
                loc.latitude,
                loc.longitude,
                (t.current_listing_price / NULLIF(li.construction_area_m, 0)) AS precio_m2
            FROM listings l
            JOIN listing_transaction_types ltt ON l.transaction_type_id = ltt.id
            JOIN listings_information li       ON li.listing_id = l.id
            JOIN locations loc                 ON loc.listing_id = l.id
            JOIN cities ci                     ON loc.city_id = ci.id
            JOIN transactions t ON t.listing_id = l.id
                AND t.transaction_type_id = 2
                AND t.transaction_status_id IN (2, 5)
            WHERE ltt.id = 2
              AND l.status_listing_id = 7
              AND ci.name = :ciudad
              AND loc.latitude  BETWEEN :lat_min AND :lat_max
              AND loc.longitude BETWEEN :lng_min AND :lng_max
              AND t.sold_date >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
              AND li.construction_area_m > 0
              AND t.current_listing_price > 0
        """)

        try:
            with crm_engine.connect() as conn:
                df_estado_venta = pd.read_sql(ratio_venta_query, conn, params=params)
                df_precios_venta = pd.read_sql(promedio_venta_query, conn, params=params)
                df_estado_alquiler = pd.read_sql(ratio_alquiler_query, conn, params=params)
                df_precios_alquiler = pd.read_sql(promedio_alquiler_query, conn, params=params)
        except Exception as e:
            log.error(f"  Error calculando features para {ciudad}: {e}")
            continue

        # ── Procesar VENTA ───────────────────────────────────────────────
        resumen_ratio_venta = _calcular_ratio(
            df_estado_venta, ciudad, "Venta Aceptada/Vendida", asignar_cluster_a
        )
        resumen_precios_venta = _calcular_promedio_m2(df_precios_venta, ciudad, asignar_cluster_a)

        # ── Procesar ALQUILER ────────────────────────────────────────────
        resumen_ratio_alquiler = _calcular_ratio(
            df_estado_alquiler, ciudad, "Alquilado", asignar_cluster_a
        )
        resumen_precios_alquiler = _calcular_promedio_m2(
            df_precios_alquiler, ciudad, asignar_cluster_a
        )

        # ── Aplicar a filas de esta ciudad según tipo_transaccion ────────
        mask_ciudad = df["ciudad"] == ciudad
        mask_venta = mask_ciudad & (df["tipo_transaccion"] == "Venta")
        mask_alquiler = mask_ciudad & (df["tipo_transaccion"] == "Alquiler")

        df = _aplicar_features_zona(df, mask_venta, resumen_ratio_venta, resumen_precios_venta)

        df = _aplicar_features_zona(
            df, mask_alquiler, resumen_ratio_alquiler, resumen_precios_alquiler
        )

        n_venta = mask_venta.sum()
        n_alquiler = mask_alquiler.sum()
        log.info(f"  ✓ {ciudad}: {n_venta} ventas + {n_alquiler} alquileres con features de zona")

    return df


def _calcular_ratio(
    df_estado: pd.DataFrame, ciudad: str, status_cerrado: str, asignar_fn
) -> pd.DataFrame:
    """Calcula ratio activas/cerradas por cluster."""
    if len(df_estado) == 0:
        return pd.DataFrame(columns=["cluster_zona", "ratio_activas_vendidas_zona"])

    df_estado = asignar_fn(df_estado, ciudad)

    resumen = (
        df_estado.assign(
            activas=df_estado["status"].eq("Activa").astype(int),
            cerradas=df_estado["status"].eq(status_cerrado).astype(int),
        )
        .groupby("cluster_zona", as_index=False)
        .agg(activas=("activas", "sum"), cerradas=("cerradas", "sum"))
    )

    resumen["ratio_activas_vendidas_zona"] = resumen["activas"] / resumen["cerradas"].replace(0, 1)
    return resumen[["cluster_zona", "ratio_activas_vendidas_zona"]]


def _calcular_promedio_m2(df_precios: pd.DataFrame, ciudad: str, asignar_fn) -> pd.DataFrame:
    """Calcula promedio precio_m2 por cluster."""
    if len(df_precios) == 0:
        return pd.DataFrame(columns=["cluster_zona", "promedio_m2_zona"])

    df_precios = asignar_fn(df_precios, ciudad)

    return (
        df_precios.groupby("cluster_zona").agg(promedio_m2_zona=("precio_m2", "mean")).reset_index()
    )


def _aplicar_features_zona(
    df: pd.DataFrame,
    mask: pd.Series,
    resumen_ratio: pd.DataFrame,
    resumen_precios: pd.DataFrame,
) -> pd.DataFrame:
    """Aplica ratio y diferencia_vs_promedio a las filas del mask."""
    if not mask.any():
        return df

    clusters = df.loc[mask, "cluster_zona"]

    if not resumen_ratio.empty:
        ratio_por_cluster = resumen_ratio.set_index("cluster_zona")["ratio_activas_vendidas_zona"]
        df.loc[mask, "ratio_activas_vendidas_zona"] = (
            clusters.map(ratio_por_cluster).fillna(0).to_numpy()
        )

    if not resumen_precios.empty:
        promedio_por_cluster = resumen_precios.set_index("cluster_zona")["promedio_m2_zona"]
        promedio = clusters.map(promedio_por_cluster)
        precios = pd.to_numeric(df.loc[mask, "precio_m2"], errors="coerce")
        diferencia = ((precios - promedio) / promedio).where(promedio.gt(0), 0)
        df.loc[mask, "diferencia_vs_promedio_zona"] = diferencia.fillna(0).to_numpy()

    return df
