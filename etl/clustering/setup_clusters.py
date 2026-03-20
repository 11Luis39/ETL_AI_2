import logging
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Configuración de ciudades
# ------------------------------------------------------------
CIUDADES_CONFIG = {
    # ── Tier 1 — Clusters propios ──────────────────────────
    "Santa Cruz de la Sierra": {
        "tier":       1,
        "n_clusters": 10,
        "bbox": {"lat": (-18.5, -17.0), "lng": (-64.0, -62.5)},
    },
    "Cochabamba": {
        "tier":       1,
        "n_clusters": 6,
        "bbox": {"lat": (-17.8, -17.2), "lng": (-66.5, -65.8)},
    },
    "La Paz": {
        "tier":       1,
        "n_clusters": 5,
        "bbox": {"lat": (-16.8, -16.3), "lng": (-68.3, -67.8)},
    },
    # ── Tier 2 — Cluster único ─────────────────────────────
    "Porongo":     {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-17.9, -17.5), "lng": (-63.6, -63.2)}},
    "El Alto":     {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-16.6, -16.4), "lng": (-68.3, -68.0)}},
    "Oruro":       {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-18.1, -17.8), "lng": (-67.2, -67.0)}},
    "Tiquipaya":   {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-17.4, -17.2), "lng": (-66.3, -66.1)}},
    "Sacaba":      {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-17.5, -17.3), "lng": (-65.9, -65.7)}},
    "Sucre":       {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-19.2, -18.9), "lng": (-65.4, -65.1)}},
    "La Guardia":  {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-17.9, -17.7), "lng": (-63.4, -63.2)}},
    "Warnes":      {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-17.6, -17.4), "lng": (-63.3, -63.1)}},
    "Quillacollo": {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-17.5, -17.3), "lng": (-66.3, -66.1)}},
    "Samaipata":   {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-18.3, -18.1), "lng": (-63.9, -63.7)}},
    "Cotoca":      {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-17.9, -17.7), "lng": (-63.1, -62.9)}},
    "Potosí":      {"tier": 2, "n_clusters": 1, "bbox": {"lat": (-19.7, -19.4), "lng": (-65.9, -65.6)}},
}

# ------------------------------------------------------------
# Conexiones
# ------------------------------------------------------------
def get_crm_engine():
    url = (
        f"mysql+pymysql://{os.getenv('CRM_USERNAME')}:{os.getenv('CRM_PASSWORD')}"
        f"@{os.getenv('CRM_HOST')}:{os.getenv('CRM_PORT')}/{os.getenv('CRM_DATABASE')}"
    )
    return create_engine(url)

def get_pg_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    )
    return create_engine(url)

# ------------------------------------------------------------
# Extraer coordenadas — ventas + alquileres
# Usamos toda la actividad para que los clusters representen
# la geografía real del mercado, no solo un tipo de transacción
# ------------------------------------------------------------
def extraer_coordenadas_ciudad(ciudad: str, bbox: dict, crm_engine) -> pd.DataFrame:
    query = text("""
        SELECT DISTINCT
            l.id        AS id_propiedad,
            loc.latitude,
            loc.longitude,
            ltt.name    AS tipo_transaccion
        FROM listings l
        JOIN locations loc ON loc.listing_id = l.id
        JOIN cities ci ON loc.city_id = ci.id
        JOIN listing_transaction_types ltt ON l.transaction_type_id = ltt.id
        JOIN transactions t ON t.listing_id = l.id
            AND t.transaction_status_id IN (2, 5)
        WHERE ltt.id IN (1, 2)
          AND l.status_listing_id IN (7, 8)
          AND ci.name = :ciudad
          AND loc.latitude  BETWEEN :lat_min AND :lat_max
          AND loc.longitude BETWEEN :lng_min AND :lng_max
          AND YEAR(t.sold_date) IN (2025, 2026)
    """)
    with crm_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "ciudad":  ciudad,
            "lat_min": bbox["lat"][0],
            "lat_max": bbox["lat"][1],
            "lng_min": bbox["lng"][0],
            "lng_max": bbox["lng"][1],
        })

    ventas     = (df["tipo_transaccion"] == "Venta").sum()
    alquileres = (df["tipo_transaccion"] == "Alquiler").sum()
    log.info(f"  → {len(df)} propiedades ({ventas} ventas + {alquileres} alquileres)")
    return df

# ------------------------------------------------------------
# Elbow Method — solo Tier 1
# ------------------------------------------------------------
def elbow_method(coords: np.ndarray, ciudad: str, max_k: int = 20):
    log.info(f"  Corriendo Elbow Method para {ciudad}...")
    inercias  = []
    valores_k = list(range(2, min(max_k, len(coords) // 10) + 1, 2))

    for k in valores_k:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(coords)
        inercias.append(modelo.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(valores_k, inercias, marker="o", linewidth=2, color="#2563EB")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.title(f"Elbow Method — {ciudad}")
    plt.xticks(valores_k)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("data", exist_ok=True)
    nombre = ciudad.lower().replace(" ", "_").replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
    ruta = f"data/elbow_{nombre}.png"
    plt.savefig(ruta)
    plt.close()
    log.info(f"  Gráfica guardada en {ruta}")
    return valores_k, inercias

# ------------------------------------------------------------
# Guardar centroides en PostgreSQL
# ------------------------------------------------------------
def guardar_centroides(modelo: KMeans, ciudad: str, n_clusters: int, pg_engine):
    centroides = modelo.cluster_centers_
    labels     = modelo.labels_
    conteo     = pd.Series(labels).value_counts().to_dict()

    insert_sql = text("""
        INSERT INTO zona_clusters
            (cluster_id, ciudad, pais, centroide_lat, centroide_lng,
             total_propiedades, n_clusters_ciudad)
        VALUES
            (:cluster_id, :ciudad, :pais, :centroide_lat, :centroide_lng,
             :total_propiedades, :n_clusters_ciudad)
        ON CONFLICT (cluster_id, ciudad) DO UPDATE SET
            centroide_lat      = EXCLUDED.centroide_lat,
            centroide_lng      = EXCLUDED.centroide_lng,
            total_propiedades  = EXCLUDED.total_propiedades,
            n_clusters_ciudad  = EXCLUDED.n_clusters_ciudad,
            fecha_generacion   = NOW()
    """)

    with pg_engine.begin() as conn:
        conn.execute(
            text("DELETE FROM zona_clusters WHERE ciudad = :ciudad"),
            {"ciudad": ciudad}
        )
        for i, centroide in enumerate(centroides):
            conn.execute(insert_sql, {
                "cluster_id":        i,
                "ciudad":            ciudad,
                "pais":              "Bolivia",
                "centroide_lat":     float(centroide[0]),
                "centroide_lng":     float(centroide[1]),
                "total_propiedades": int(conteo.get(i, 0)),
                "n_clusters_ciudad": n_clusters,
            })

    log.info(f"  ✓ {n_clusters} clusters guardados para {ciudad}")

# ------------------------------------------------------------
# Procesar una ciudad
# ------------------------------------------------------------
def procesar_ciudad(ciudad: str, config: dict, crm_engine, pg_engine):
    tier = config["tier"]
    bbox = config["bbox"]

    log.info(f"\n{'='*55}")
    log.info(f"  {ciudad}  [Tier {tier}]")
    log.info(f"{'='*55}")

    df = extraer_coordenadas_ciudad(ciudad, bbox, crm_engine)

    if len(df) < 5:
        log.warning(f"  ⚠️  Menos de 5 propiedades — saltando {ciudad}")
        return None

    coords = df[["latitude", "longitude"]].values

    if tier == 1:
        elbow_method(coords, ciudad)
        max_posible = min(config["n_clusters"] + 5, len(df) // 5)
        nombre = ciudad.lower().replace(" ", "_").replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
        print(f"\n  Revisá data/elbow_{nombre}.png")
        print(f"  Sugerido: {config['n_clusters']} | Máximo recomendado: {max_posible}")
        n_clusters = int(input(f"  ¿Cuántos clusters para {ciudad}? "))
    else:
        n_clusters = 1
        log.info(f"  → Tier 2: usando 1 cluster (centroide)")

    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    modelo.fit(coords)
    guardar_centroides(modelo, ciudad, n_clusters, pg_engine)

    return {
        "ciudad":      ciudad,
        "tier":        tier,
        "propiedades": len(df),
        "clusters":    n_clusters,
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("INTRAMAX — Setup Clusters Multiciudad Bolivia")
    log.info("  Usando ventas + alquileres 2025/2026")
    log.info("=" * 55)

    crm_engine = get_crm_engine()
    pg_engine  = get_pg_engine()

    resumen = []
    for ciudad, config in CIUDADES_CONFIG.items():
        resultado = procesar_ciudad(ciudad, config, crm_engine, pg_engine)
        if resultado:
            resumen.append(resultado)

    log.info(f"\n{'='*55}")
    log.info("RESUMEN FINAL")
    log.info(f"{'='*55}")
    log.info(f"  {'Ciudad':30} {'Props':>6}  {'Clusters':>8}  Tier")
    log.info(f"  {'-'*50}")
    for r in resumen:
        log.info(f"  {r['ciudad']:30} {r['propiedades']:>6}  {r['clusters']:>8}  T{r['tier']}")
    log.info(f"\n  Total ciudades procesadas: {len(resumen)}")
    log.info(f"  Ya podés correr: python etl/main.py")