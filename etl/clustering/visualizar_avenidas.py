import folium
from folium.plugins import MiniMap, Fullscreen
import geopandas as gpd
import pandas as pd
import os
import webbrowser
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Paleta de colores por categoría (debe coincidir con setup_avenidas.py)
COLORES = {
    "conexion_nacional":   "#DC2626",
    "anillos_principales": "#EA580C",
    "anillos_expansion":   "#F59E0B",
    "primarias":           "#2563EB",
    "plan_3000":           "#7C3AED",
    "villa":               "#059669",
    "pampa":               "#0891B2",
    "zona_sur":            "#BE185D",
    "alimentadoras":       "#65A30D",
    "casco_viejo":         "#92400E",
    "expansion":           "#6B7280",
}

LABELS = {
    "conexion_nacional":   "🛣️  Conexión Nacional/Regional",
    "anillos_principales": "⭕ Anillos Principales (1ro–4to)",
    "anillos_expansion":   "⭕ Anillos Expansión (5to–8vo)",
    "primarias":           "🔵 Avenidas Primarias",
    "plan_3000":           "🟣 Plan 3.000 (D-8)",
    "villa":               "🟢 Villa 1ro de Mayo (D-7)",
    "pampa":               "🔵 Pampa de la Isla (D-6)",
    "zona_sur":            "🩷 Los Lotes / Zona Sur",
    "alimentadoras":       "🟡 Alimentadoras Importantes",
    "casco_viejo":         "🟤 Casco Viejo / Alto Valor",
    "expansion":           "⚪ Vías de Expansión",
}


def cargar_clusters():
    """Carga los clusters de zonas desde PostgreSQL"""
    try:
        from sqlalchemy import create_engine
        import os as _os
        url = (
            f"postgresql+psycopg2://{_os.getenv('PG_USERNAME')}:{_os.getenv('PG_PASSWORD')}"
            f"@{_os.getenv('PG_HOST')}:{_os.getenv('PG_PORT')}/{_os.getenv('PG_DATABASE')}"
        )
        engine = create_engine(url)
        with engine.connect() as conn:
            clusters = pd.read_sql(
                "SELECT cluster_id, centroide_lat, centroide_lng, total_propiedades FROM zona_clusters",
                conn
            )
        log.info(f"  → {len(clusters)} clusters cargados desde PostgreSQL")
        return clusters
    except Exception as e:
        log.warning(f"  No se pudieron cargar clusters: {e}")
        return None


def generar_leyenda_html(categorias_en_mapa: list) -> str:
    """Genera el HTML de la leyenda lateral"""
    items = ""
    for cat in categorias_en_mapa:
        color = COLORES.get(cat, "#6B7280")
        label = LABELS.get(cat, cat)
        items += f"""
        <div style="display:flex; align-items:center; margin-bottom:6px;">
            <div style="width:28px; height:4px; background:{color};
                        border-radius:2px; margin-right:8px; flex-shrink:0;"></div>
            <span style="font-size:12px; color:#374151;">{label}</span>
        </div>"""

    return f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 1000;
        background: white;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        font-family: 'Segoe UI', sans-serif;
        min-width: 260px;
        border: 1px solid #E5E7EB;
    ">
        <div style="font-weight:700; font-size:13px; color:#111827;
                    margin-bottom:12px; padding-bottom:8px;
                    border-bottom:2px solid #F3F4F6;">
            🗺️ Red Vial — Santa Cruz de la Sierra
        </div>
        {items}
        <div style="margin-top:12px; padding-top:8px; border-top:1px solid #F3F4F6;
                    font-size:11px; color:#9CA3AF;">
            Fuente: OpenStreetMap © 2025
        </div>
    </div>
    """


def generar_mapa():
    ruta_vias = "data/geo/vias_clasificadas.gpkg"

    if not os.path.exists(ruta_vias):
        log.error("No se encontró data/geo/vias_clasificadas.gpkg")
        log.error("Primero corrí: python etl/clustering/setup_avenidas.py")
        return

    log.info("Cargando vías clasificadas...")
    vias = gpd.read_file(ruta_vias)
    log.info(f"  → {len(vias):,} segmentos cargados")

    # Convertir a WGS84
    if vias.crs and vias.crs.to_epsg() != 4326:
        vias = vias.to_crs("EPSG:4326")

    # ------------------------------------------------------------
    # Crear mapa base
    # ------------------------------------------------------------
    mapa = folium.Map(
        location=[-17.7833, -63.1821],
        zoom_start=12,
        tiles=None,
    )

    # Capas base
    folium.TileLayer(
        "CartoDB positron",
        name="Mapa claro",
        control=True,
    ).add_to(mapa)

    folium.TileLayer(
        "CartoDB dark_matter",
        name="Mapa oscuro",
        control=True,
    ).add_to(mapa)

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(mapa)

    # ------------------------------------------------------------
    # Agregar clusters de zonas (si están disponibles)
    # ------------------------------------------------------------
    clusters = cargar_clusters()
    if clusters is not None:
        grupo_clusters = folium.FeatureGroup(name="🏘️ Zonas de Propiedades", show=True)
        COLORES_CLUSTER = [
            "#EF4444", "#3B82F6", "#10B981", "#8B5CF6", "#F59E0B",
            "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1"
        ]
        for _, row in clusters.iterrows():
            cid   = int(row["cluster_id"])
            color = COLORES_CLUSTER[cid % len(COLORES_CLUSTER)]
            folium.CircleMarker(
                location=[row["centroide_lat"], row["centroide_lng"]],
                radius=18,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.25,
                weight=2,
                tooltip=folium.Tooltip(
                    f"<b>Zona {cid}</b><br>{int(row['total_propiedades'])} propiedades",
                    style="font-family: Segoe UI; font-size: 13px;"
                ),
            ).add_to(grupo_clusters)

            folium.Marker(
                location=[row["centroide_lat"], row["centroide_lng"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:11px; font-weight:bold; color:{color}; '
                         f'text-shadow: 1px 1px 2px white, -1px -1px 2px white;">{cid}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                ),
            ).add_to(grupo_clusters)

        grupo_clusters.add_to(mapa)

    # ------------------------------------------------------------
    # Dibujar vías por categoría (una FeatureGroup por categoría)
    # ------------------------------------------------------------
    categorias_en_mapa = []
    segmentos_totales  = 0

    for categoria in COLORES.keys():
        df_cat = vias[vias["categoria"] == categoria]
        if len(df_cat) == 0:
            continue

        color  = COLORES[categoria]
        label  = LABELS.get(categoria, categoria)
        weight = int(df_cat["weight"].iloc[0]) if "weight" in df_cat.columns else 2

        grupo = folium.FeatureGroup(name=label, show=True)

        for _, row in df_cat.iterrows():
            geom     = row.geometry
            nombre   = str(row.get("nombre_via", ""))
            cat_label = str(row.get("label", ""))

            tooltip_text = f"<b>{nombre}</b><br><span style='color:{color}'>{cat_label}</span>"

            try:
                if geom.geom_type == "LineString":
                    coords = [[pt[1], pt[0]] for pt in geom.coords]
                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=weight,
                        opacity=0.85,
                        tooltip=folium.Tooltip(
                            tooltip_text,
                            style="font-family: Segoe UI; font-size: 12px;"
                        ),
                    ).add_to(grupo)
                    segmentos_totales += 1

                elif geom.geom_type == "MultiLineString":
                    for line in geom.geoms:
                        coords = [[pt[1], pt[0]] for pt in line.coords]
                        folium.PolyLine(
                            coords,
                            color=color,
                            weight=weight,
                            opacity=0.85,
                            tooltip=folium.Tooltip(
                                tooltip_text,
                                style="font-family: Segoe UI; font-size: 12px;"
                            ),
                        ).add_to(grupo)
                        segmentos_totales += 1
            except Exception:
                continue

        grupo.add_to(mapa)
        categorias_en_mapa.append(categoria)
        log.info(f"  → {label}: {len(df_cat)} segmentos dibujados")

    log.info(f"\n  Total segmentos dibujados: {segmentos_totales:,}")

    # ------------------------------------------------------------
    # Controles y extras
    # ------------------------------------------------------------
    folium.LayerControl(collapsed=False, position="topright").add_to(mapa)
    Fullscreen(position="topleft").add_to(mapa)
    MiniMap(toggle_display=True, position="bottomright").add_to(mapa)

    # Leyenda
    leyenda = generar_leyenda_html(categorias_en_mapa)
    mapa.get_root().html.add_child(folium.Element(leyenda))

    # Título
    titulo_html = """
    <div style="
        position: fixed;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        background: white;
        padding: 10px 24px;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        font-weight: 700;
        color: #111827;
        border: 1px solid #E5E7EB;
        white-space: nowrap;
    ">
        INTRAMAX · Red Vial Estratégica — Santa Cruz de la Sierra
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(titulo_html))

    # ------------------------------------------------------------
    # Guardar y abrir
    # ------------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    ruta_salida = "data/mapa_avenidas.html"
    mapa.save(ruta_salida)

    ruta_abs = os.path.abspath(ruta_salida)
    log.info(f"\n✓ Mapa guardado en: {ruta_salida}")
    log.info(f"  Abriendo en navegador...")
    webbrowser.open(f"file:///{ruta_abs}")


if __name__ == "__main__":
    log.info("="*60)
    log.info("INTRAMAX — Visualizador Red Vial Santa Cruz")
    log.info("="*60)
    generar_mapa()