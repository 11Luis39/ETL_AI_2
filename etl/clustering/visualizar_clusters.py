import folium
from folium.plugins import MiniMap, Fullscreen
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics import pairwise_distances
from dotenv import load_dotenv
import os
import webbrowser
import logging
import sys

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

COLORES_HEX = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FFE119", "#FABEBE",
]

COLORES_ICONOS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue", "lightred",
]

BBOX_CIUDAD = {
    "Santa Cruz de la Sierra": {"lat": (-18.5, -17.0), "lng": (-64.0, -62.5)},
    "Cochabamba":              {"lat": (-17.8, -17.2), "lng": (-66.5, -65.8)},
    "La Paz":                  {"lat": (-16.8, -16.3), "lng": (-68.3, -67.8)},
    "Porongo":                 {"lat": (-17.9, -17.5), "lng": (-63.6, -63.2)},
    "El Alto":                 {"lat": (-16.6, -16.4), "lng": (-68.3, -68.0)},
    "Oruro":                   {"lat": (-18.1, -17.8), "lng": (-67.2, -67.0)},
    "Tiquipaya":               {"lat": (-17.4, -17.2), "lng": (-66.3, -66.1)},
    "Sacaba":                  {"lat": (-17.5, -17.3), "lng": (-65.9, -65.7)},
    "Sucre":                   {"lat": (-19.2, -18.9), "lng": (-65.4, -65.1)},
    "La Guardia":              {"lat": (-17.9, -17.7), "lng": (-63.4, -63.2)},
    "Warnes":                  {"lat": (-17.6, -17.4), "lng": (-63.3, -63.1)},
    "Quillacollo":             {"lat": (-17.5, -17.3), "lng": (-66.3, -66.1)},
    "Samaipata":               {"lat": (-18.3, -18.1), "lng": (-63.9, -63.7)},
    "Cotoca":                  {"lat": (-17.9, -17.7), "lng": (-63.1, -62.9)},
    "Potosí":                  {"lat": (-19.7, -19.4), "lng": (-65.9, -65.6)},
}

# Colores para diferenciar ventas y alquileres en el mapa
COLOR_VENTA    = "#2563EB"   # azul
COLOR_ALQUILER = "#16A34A"   # verde

def get_pg_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    )
    return create_engine(url)

def get_crm_engine():
    url = (
        f"mysql+pymysql://{os.getenv('CRM_USERNAME')}:{os.getenv('CRM_PASSWORD')}"
        f"@{os.getenv('CRM_HOST')}:{os.getenv('CRM_PORT')}/{os.getenv('CRM_DATABASE')}"
    )
    return create_engine(url)

# ------------------------------------------------------------
# Cargar centroides
# ------------------------------------------------------------
def cargar_centroides() -> pd.DataFrame:
    engine = get_pg_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            "SELECT * FROM zona_clusters ORDER BY ciudad, cluster_id", conn
        )
    ciudades = df["ciudad"].unique().tolist()
    log.info(f"  → {len(df)} clusters cargados para {len(ciudades)} ciudades")
    for ciudad in ciudades:
        n = len(df[df["ciudad"] == ciudad])
        log.info(f"     {ciudad}: {n} cluster(s)")
    return df

# ------------------------------------------------------------
# Cargar propiedades — ventas Y alquileres por ciudad
# ------------------------------------------------------------
def cargar_propiedades_ciudad(ciudad: str, bbox: dict, crm_engine) -> pd.DataFrame:
    query = text("""
        SELECT DISTINCT
            loc.latitude,
            loc.longitude,
            ltt.name AS tipo_transaccion
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
    df["ciudad"] = ciudad
    return df

# ------------------------------------------------------------
# Asignar cluster por ciudad
# ------------------------------------------------------------
def asignar_clusters(df_props: pd.DataFrame, df_centroides: pd.DataFrame) -> pd.DataFrame:
    resultados = []
    for ciudad in df_props["ciudad"].unique():
        props_ciudad = df_props[df_props["ciudad"] == ciudad].copy()
        cents_ciudad = df_centroides[df_centroides["ciudad"] == ciudad]

        if len(cents_ciudad) == 0:
            log.warning(f"  No hay centroides para {ciudad} — saltando")
            continue

        coords     = props_ciudad[["latitude", "longitude"]].values
        centroides = cents_ciudad[["centroide_lat", "centroide_lng"]].values
        distancias = pairwise_distances(coords, centroides, metric="euclidean")
        indices    = distancias.argmin(axis=1)

        props_ciudad["cluster_id"] = cents_ciudad["cluster_id"].values[indices]
        resultados.append(props_ciudad)

    return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()

# ------------------------------------------------------------
# Generar mapa
# ------------------------------------------------------------
def generar_mapa(df_centroides: pd.DataFrame, df_props: pd.DataFrame) -> folium.Map:
    mapa = folium.Map(location=[-16.5, -64.5], zoom_start=6, tiles=None)
    folium.TileLayer("CartoDB positron",    name="Mapa claro").add_to(mapa)
    folium.TileLayer("CartoDB dark_matter", name="Mapa oscuro").add_to(mapa)
    folium.TileLayer(
    tiles="https://tile.opentopomap.org/{z}/{x}/{y}.png",
    attr="OpenTopoMap",
    name="Topográfico",
    ).add_to(mapa)

    folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri Street Map",
    ).add_to(mapa)

    ciudades = df_centroides["ciudad"].unique().tolist()

    for ciudad in ciudades:
        cents_ciudad = df_centroides[df_centroides["ciudad"] == ciudad]
        props_ciudad = df_props[df_props["ciudad"] == ciudad] if len(df_props) > 0 else pd.DataFrame()
        n_clusters   = len(cents_ciudad)

        ventas_n     = (props_ciudad["tipo_transaccion"] == "Venta").sum()    if len(props_ciudad) > 0 else 0
        alquileres_n = (props_ciudad["tipo_transaccion"] == "Alquiler").sum() if len(props_ciudad) > 0 else 0

        grupo = folium.FeatureGroup(
            name=f"📍 {ciudad} ({n_clusters} zonas | {ventas_n}V + {alquileres_n}A)",
            show=True
        )

        # Puntos — color según tipo de transacción
        if len(props_ciudad) > 0:
            for _, row in props_ciudad.iterrows():
                es_alquiler = row.get("tipo_transaccion") == "Alquiler"
                cid         = int(row["cluster_id"])
                color_base  = COLORES_HEX[cid % len(COLORES_HEX)]

                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3,
                    color=color_base,
                    fill=True,
                    fill_color=color_base,
                    fill_opacity=0.6,
                    weight=0.5,
                    tooltip=f"{'Alquiler' if es_alquiler else 'Venta'} — Zona {cid}",
                ).add_to(grupo)

        # Centroides
        for _, row in cents_ciudad.iterrows():
            cid        = int(row["cluster_id"])
            color_icon = COLORES_ICONOS[cid % len(COLORES_ICONOS)]
            total      = int(row["total_propiedades"])

            folium.Marker(
                location=[row["centroide_lat"], row["centroide_lng"]],
                popup=folium.Popup(
                    f"<b>{ciudad}</b><br>Zona {cid}<br>Propiedades: {total}",
                    max_width=200
                ),
                tooltip=f"{ciudad} — Zona {cid} ({total} props)",
                icon=folium.Icon(color=color_icon, icon="home"),
            ).add_to(grupo)

        grupo.add_to(mapa)
        log.info(f"  → {ciudad}: {ventas_n} ventas + {alquileres_n} alquileres, {n_clusters} zonas")

    # Leyenda tipo transacción
    leyenda = f"""
    <div style="position:fixed; bottom:80px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                box-shadow:0 2px 10px rgba(0,0,0,0.15);
                font-family:'Segoe UI',sans-serif; font-size:12px;">
        <b>Tipo de transacción</b><br><br>
        <span style="color:{COLOR_VENTA}">●</span> Venta<br>
        <span style="color:{COLOR_ALQUILER}">●</span> Alquiler
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(leyenda))

    folium.LayerControl(collapsed=False, position="topright").add_to(mapa)
    Fullscreen(position="topleft").add_to(mapa)
    MiniMap(toggle_display=True, position="bottomright").add_to(mapa)

    titulo = """
    <div style="position:fixed; top:15px; left:50%; transform:translateX(-50%);
                z-index:1000; background:white; padding:10px 24px;
                border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.12);
                font-family:'Segoe UI',sans-serif; font-size:14px;
                font-weight:700; color:#111827; border:1px solid #E5E7EB;
                white-space:nowrap;">
        INTRAMAX · Zonas de Propiedades — Bolivia (Ventas + Alquileres)
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(titulo))
    return mapa

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    log.info("="*55)
    log.info("INTRAMAX — Visualizador Clusters Multiciudad")
    log.info("="*55)

    crm_engine = get_crm_engine()

    log.info("\nCargando centroides desde PostgreSQL...")
    df_centroides = cargar_centroides()

    if df_centroides.empty:
        log.error("No hay clusters. Corré setup_clusters.py primero.")
        sys.exit(1)

    log.info("\nCargando propiedades desde CRM...")
    dfs_props = []
    for ciudad in df_centroides["ciudad"].unique():
        if ciudad in BBOX_CIUDAD:
            df_p = cargar_propiedades_ciudad(ciudad, BBOX_CIUDAD[ciudad], crm_engine)
            log.info(f"  → {ciudad}: {len(df_p)} propiedades")
            dfs_props.append(df_p)
        else:
            log.warning(f"  Sin bbox para {ciudad} — saltando")

    if dfs_props:
        df_props = pd.concat(dfs_props, ignore_index=True)
        df_props = asignar_clusters(df_props, df_centroides)
        log.info(f"\n  Total: {len(df_props)} propiedades")
    else:
        df_props = pd.DataFrame()

    log.info("\nGenerando mapa...")
    mapa = generar_mapa(df_centroides, df_props)

    os.makedirs("data", exist_ok=True)
    ruta = "data/mapa_clusters.html"
    mapa.save(ruta)
    log.info(f"\n✓ Mapa guardado en {ruta}")
    webbrowser.open(f"file:///{os.path.abspath(ruta)}")