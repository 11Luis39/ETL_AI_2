import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import webbrowser
import logging
import sys
import json

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

def get_pg_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    )
    return create_engine(url)

# ------------------------------------------------------------
# Cargar datos
# ------------------------------------------------------------
def cargar_datos(ciudad: str = None) -> pd.DataFrame:
    engine = get_pg_engine()
    if ciudad:
        query = text("""
            SELECT id_propiedad, mlsid, tipo_propiedad, tipo_transaccion,
                   cluster_zona, ciudad, latitude, longitude,
                   precio_venta, precio_alquiler_mes, precio_m2,
                   m2_construidos, m2_terreno, dormitorios, banos,
                   fecha_venta,
                   EXTRACT(YEAR  FROM fecha_venta) AS anio_venta,
                   EXTRACT(MONTH FROM fecha_venta) AS mes_venta
            FROM property_analytics
            WHERE ciudad = :ciudad
              AND latitude IS NOT NULL AND longitude IS NOT NULL
              AND (precio_venta IS NOT NULL OR precio_alquiler_mes IS NOT NULL)
            ORDER BY fecha_venta DESC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"ciudad": ciudad})
    else:
        query = text("""
            SELECT id_propiedad, mlsid, tipo_propiedad, tipo_transaccion,
                   cluster_zona, ciudad, latitude, longitude,
                   precio_venta, precio_alquiler_mes, precio_m2,
                   m2_construidos, m2_terreno, dormitorios, banos,
                   fecha_venta,
                   EXTRACT(YEAR  FROM fecha_venta) AS anio_venta,
                   EXTRACT(MONTH FROM fecha_venta) AS mes_venta
            FROM property_analytics
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND (precio_venta IS NOT NULL OR precio_alquiler_mes IS NOT NULL)
            ORDER BY fecha_venta DESC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

    log.info(f"  → {len(df)} propiedades cargadas{' para ' + ciudad if ciudad else ' (Bolivia completo)'}")
    return df

def cargar_centroides(ciudad: str = None) -> pd.DataFrame:
    engine = get_pg_engine()
    if ciudad:
        query = text("""
            SELECT cluster_id, ciudad, centroide_lat, centroide_lng, total_propiedades
            FROM zona_clusters WHERE ciudad = :ciudad ORDER BY cluster_id
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"ciudad": ciudad})
    else:
        query = text("""
            SELECT cluster_id, ciudad, centroide_lat, centroide_lng, total_propiedades
            FROM zona_clusters ORDER BY ciudad, cluster_id
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    return df

# ------------------------------------------------------------
# Calcular polígonos Voronoi para los clusters
# ------------------------------------------------------------
def calcular_zonas_organicas(df_propiedades: pd.DataFrame, centroides_df: pd.DataFrame) -> list:
    """
    Crea polígonos que rodean las propiedades reales de cada cluster
    en lugar de dividir todo el mapa matemáticamente.
    """
    poligonos = []
    
    for _, centroide in centroides_df.iterrows():
        c_id = centroide["cluster_id"]
        ciudad = centroide["ciudad"]
        
        # Filtramos puntos que pertenecen a este cluster
        puntos_cluster = df_propiedades[
            (df_propiedades["cluster_zona"] == c_id) & 
            (df_propiedades["ciudad"] == ciudad)
        ][["latitude", "longitude"]].values
        
        if len(puntos_cluster) < 3:
            continue

        # En lugar de Voronoi, usamos el Convex Hull (Envolvente Convexa)
        # Es mucho más limpio visualmente para delimitar zonas de mercado.
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(puntos_cluster)
            vertices = puntos_cluster[hull.vertices].tolist()
            
            poligonos.append({
                "cluster_id": int(c_id),
                "ciudad": str(ciudad),
                "coords": vertices,
            })
        except:
            continue
            
    return poligonos
# ------------------------------------------------------------
# Preparar registros JSON
# ------------------------------------------------------------
def preparar_registros(df: pd.DataFrame) -> list:
    registros = []
    for _, row in df.iterrows():
        precio_val = row.get("precio_venta")
        if pd.isna(precio_val) or precio_val == 0:
            precio_val = row.get("precio_alquiler_mes")
        if pd.isnull(precio_val) or precio_val == 0:
            continue

        registros.append({
            "lat":          float(row["latitude"]),
            "lng":          float(row["longitude"]),
            "mlsid":        str(row["mlsid"]) if pd.notnull(row["mlsid"]) else "N/A",
            "precio":       float(precio_val),
            "precio_m2":    float(row["precio_m2"]) if pd.notnull(row["precio_m2"]) else 0,
            "tipo":         str(row["tipo_propiedad"] or ""),
            "transaccion":  str(row["tipo_transaccion"] or ""),
            "ciudad":       str(row["ciudad"] or ""),
            "cluster":      int(row["cluster_zona"]) if pd.notnull(row["cluster_zona"]) else -1,
            "anio":         int(row["anio_venta"]) if pd.notnull(row["anio_venta"]) else 0,
            "m2_construidos": float(row["m2_construidos"]) if pd.notnull(row["m2_construidos"]) else 0,
            "m2_terreno":   float(row["m2_terreno"]) if pd.notnull(row["m2_terreno"]) else 0,
            "dormitorios":  int(row["dormitorios"]) if pd.notnull(row["dormitorios"]) else 0,
            "banos":        int(row["banos"]) if pd.notnull(row["banos"]) else 0,
        })
    return registros

# ------------------------------------------------------------
# Generar mapa
# ------------------------------------------------------------
def generar_mapa(ciudad: str = None):
    modo_bolivia = ciudad is None
    titulo_ciudad = "Bolivia — Todas las ciudades" if modo_bolivia else ciudad

    log.info(f"Cargando datos{'...' if modo_bolivia else f' para {ciudad}...'}")
    df_all        = cargar_datos(ciudad)
    df_centroides = cargar_centroides(ciudad)
    poligonos_organicos = calcular_zonas_organicas(df_all, df_centroides)

    if df_all.empty:
        log.error("No hay datos. Corré el ETL primero.")
        return

    # Calcular bbox para Voronoi
    if modo_bolivia:
        bbox = {
            "lat": (df_all["latitude"].min(),  df_all["latitude"].max()),
            "lng": (df_all["longitude"].min(), df_all["longitude"].max()),
        }
        centro_lat, centro_lng = -16.5, -64.5
        zoom_inicial = 6
    else:
        bbox = {
            "lat": (df_all["latitude"].min(),  df_all["latitude"].max()),
            "lng": (df_all["longitude"].min(), df_all["longitude"].max()),
        }
        centro_lat = df_all["latitude"].mean()
        centro_lng = df_all["longitude"].mean()
        zoom_inicial = 12

    # Valores únicos para filtros
    tipos_prop  = sorted(df_all["tipo_propiedad"].dropna().unique().tolist())
    tipos_trans = sorted(df_all["tipo_transaccion"].dropna().unique().tolist())
    anios       = sorted(df_all["anio_venta"].dropna().unique().astype(int).tolist(), reverse=True)
    ciudades    = sorted(df_all["ciudad"].dropna().unique().tolist()) if modo_bolivia else []

    registros     = preparar_registros(df_all)
    centroides_js = df_centroides.to_dict("records")
    for c in centroides_js:
        c["cluster_id"]        = int(c["cluster_id"])
        c["centroide_lat"]     = float(c["centroide_lat"])
        c["centroide_lng"]     = float(c["centroide_lng"])
        c["total_propiedades"] = int(c["total_propiedades"])

    # Botones de ciudad para Bolivia completo
    botones_ciudad = ""
    if modo_bolivia:
        botones_ciudad = f"""
        <div class="filtro-grupo">
            <label>Ciudad</label>
            <div class="btn-grupo" id="filtro-ciudad">
                <button class="btn-filtro activo" data-val="todos">Todas</button>
                {''.join(f'<button class="btn-filtro" data-val="{c}">{c}</button>' for c in ciudades)}
            </div>
        </div>"""

    setup_ciudad_js = """setupFiltros("filtro-ciudad", "ciudad");""" if modo_bolivia else ""
    filtro_ciudad_init = '"ciudad": "todos",' if modo_bolivia else ""
    

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>INTRAMAX · Mapa de Precios — {titulo_ciudad}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:'Segoe UI',sans-serif; background:#f0f2f5; }}
        #app {{ display:flex; height:100vh; }}

        #panel {{
            width:300px; min-width:300px; background:white;
            box-shadow:2px 0 12px rgba(0,0,0,0.1);
            display:flex; flex-direction:column; z-index:1000; overflow-y:auto;
        }}
        #panel-header {{ background:#1e3a5f; color:white; padding:16px; }}
        #panel-header h2 {{ font-size:15px; font-weight:700; }}
        #panel-header p  {{ font-size:11px; opacity:0.8; margin-top:4px; }}

        .filtro-grupo {{ padding:12px 16px; border-bottom:1px solid #f0f0f0; }}
        .filtro-grupo label {{
            font-size:11px; font-weight:600; color:#666;
            text-transform:uppercase; letter-spacing:0.5px;
            display:block; margin-bottom:6px;
        }}
        .btn-grupo {{ display:flex; flex-wrap:wrap; gap:4px; }}
        .btn-filtro {{
            padding:4px 10px; border:1px solid #ddd; border-radius:20px;
            background:white; font-size:12px; cursor:pointer;
            transition:all 0.15s; color:#333;
        }}
        .btn-filtro:hover {{ border-color:#1e3a5f; color:#1e3a5f; }}
        .btn-filtro.activo {{ background:#1e3a5f; color:white; border-color:#1e3a5f; }}

        .vista-btn {{ display:flex; gap:4px; flex-wrap:wrap; }}
        .vista-btn button {{
            flex:1; min-width:60px; padding:6px 4px; border:1px solid #ddd;
            border-radius:6px; background:white; font-size:10px;
            cursor:pointer; transition:all 0.15s; text-align:center;
        }}
        .vista-btn button.activo {{ background:#1e3a5f; color:white; border-color:#1e3a5f; }}

        #stats-panel {{ padding:12px 16px; border-bottom:1px solid #f0f0f0; }}
        #stats-panel h3 {{
            font-size:12px; font-weight:600; color:#666;
            margin-bottom:8px; text-transform:uppercase;
        }}
        .stat-item {{
            display:flex; justify-content:space-between;
            font-size:12px; margin-bottom:4px;
        }}
        .stat-item .lbl {{ color:#888; }}
        .stat-item .val {{ font-weight:600; color:#1e3a5f; }}

        #leyenda {{ padding:12px 16px; }}
        #leyenda h3 {{
            font-size:12px; font-weight:600; color:#666;
            margin-bottom:8px; text-transform:uppercase;
        }}
        .leyenda-gradiente {{
            height:12px; border-radius:6px; margin-bottom:4px;
        }}
        .leyenda-labels {{
            display:flex; justify-content:space-between;
            font-size:10px; color:#888;
        }}
        
                /* Mejora visual de los polígonos en el mapa */
        .leaflet-interactive {{
            transition: fill-opacity 0.2s, stroke-width 0.2s;
            outline: none;
        }}

        .leaflet-interactive:hover {{
            fill-opacity: 0.6 !important;
            stroke-width: 4px !important;
        }}

        /* Panel de estadísticas más limpio */
        #stats-panel {{
            background: #f8fafc;
            margin: 10px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }}

        #map {{ flex:1; }}

        .leaflet-tooltip {{
            background:white; border:none;
            box-shadow:0 2px 8px rgba(0,0,0,0.15);
            border-radius:8px; padding:8px 12px; font-size:12px;
        }}
        .tt-titulo {{ font-weight:700; color:#1e3a5f; margin-bottom:4px; font-size:13px; }}
        .tt-fila {{ display:flex; justify-content:space-between; gap:16px; margin-bottom:2px; }}
        .tt-lbl {{ color:#888; }}
        .tt-val {{ font-weight:600; }}

        .tab-vista {{
            display:flex; border-bottom:2px solid #f0f0f0; margin-bottom:0;
        }}
        .tab-btn {{
            flex:1; padding:8px 4px; border:none; background:none;
            font-size:11px; cursor:pointer; color:#888; font-weight:600;
            border-bottom:2px solid transparent; margin-bottom:-2px;
            transition:all 0.15s;
        }}
        .tab-btn.activo {{ color:#1e3a5f; border-bottom-color:#1e3a5f; }}
    </style>
</head>
<body>
<div id="app">
<div id="panel">
    <div id="panel-header">
        <h2>🏠 Mapa de Precios INTRAMAX</h2>
        <p>{titulo_ciudad}</p>
    </div>

    <!-- Tabs de heatmap -->
    <div class="filtro-grupo" style="padding-bottom:0">
        <label>Mapa de calor</label>
        <div class="tab-vista">
            <button class="tab-btn activo" id="tab-precio"      onclick="setHeatmap('precio')">💰 Precio</button>
            <button class="tab-btn"        id="tab-densidad"    onclick="setHeatmap('densidad')">📊 Actividad</button>
        </div>
    </div>

    <!-- Vista de capa -->
    <div class="filtro-grupo">
        <label>Capa adicional</label>
        <div class="vista-btn">
            <button id="btn-ninguno"  class="activo" onclick="setCapa('ninguno')">Sin capa</button>
            <button id="btn-puntos"   onclick="setCapa('puntos')">📍 Puntos</button>
            <button id="btn-clusters" onclick="setCapa('clusters')">⭕ Zonas</button>
            <button id="btn-voronoi"  onclick="setCapa('voronoi')">🗺️ Voronoi</button>
        </div>
    </div>

    {botones_ciudad}

    <!-- Tipo transacción -->
    <div class="filtro-grupo">
        <label>Transacción</label>
        <div class="btn-grupo" id="filtro-transaccion">
            <button class="btn-filtro activo" data-val="todos">Todos</button>
            {''.join(f'<button class="btn-filtro" data-val="{t}">{t}</button>' for t in tipos_trans)}
        </div>
    </div>

    <!-- Tipo propiedad -->
    <div class="filtro-grupo">
        <label>Tipo de propiedad</label>
        <div class="btn-grupo" id="filtro-tipo">
            <button class="btn-filtro activo" data-val="todos">Todos</button>
            {''.join(f'<button class="btn-filtro" data-val="{t}">{t}</button>' for t in tipos_prop)}
        </div>
    </div>

    <!-- Año -->
    <div class="filtro-grupo">
        <label>Año</label>
        <div class="btn-grupo" id="filtro-anio">
            <button class="btn-filtro activo" data-val="0">Todos</button>
            {''.join(f'<button class="btn-filtro" data-val="{a}">{a}</button>' for a in anios)}
        </div>
    </div>

    <!-- Stats -->
    <div id="stats-panel">
        <h3>Resumen</h3>
        <div class="stat-item"><span class="lbl">Propiedades</span><span class="val" id="st-total">—</span></div>
        <div class="stat-item"><span class="lbl">Precio promedio</span><span class="val" id="st-prom">—</span></div>
        <div class="stat-item"><span class="lbl">Precio mínimo</span><span class="val" id="st-min">—</span></div>
        <div class="stat-item"><span class="lbl">Precio máximo</span><span class="val" id="st-max">—</span></div>
        <div class="stat-item"><span class="lbl">Precio/m²</span><span class="val" id="st-m2">—</span></div>
        <div class="stat-item"><span class="lbl">Operaciones</span><span class="val" id="st-ops">—</span></div>
    </div>

    <!-- Leyenda -->
    <div id="leyenda">
        <h3 id="leyenda-titulo">Escala de precios</h3>
        <div class="leyenda-gradiente" id="leyenda-grad"
             style="background:linear-gradient(to right,#00FF00,#FFFF00,#FF0000)"></div>
        <div class="leyenda-labels">
            <span id="leg-min">—</span>
            <span id="leg-mid">promedio</span>
            <span id="leg-max">—</span>
        </div>
    </div>
</div>

<div id="map"></div>
</div>

<script>
const DATOS      = {json.dumps(registros)};
const CENTROIDES = {json.dumps(centroides_js)};
const VORONOI    = {json.dumps(poligonos_organicos)}; // <-- AGREGAR ESTO
const COLORES    = {json.dumps(COLORES_HEX)};
const CENTRO     = [{centro_lat}, {centro_lng}];
const ZOOM_INIT  = {zoom_inicial};

let filtros = {{ {filtro_ciudad_init} transaccion:"todos", tipo:"todos", anio:0 }};
let modoHeatmap = "precio";
let modoCapa    = "ninguno";

let layerHeat    = null;
let layerPuntos  = null;
let layerClusters = null;
let layerVoronoi = null;

const map = L.map("map").setView(CENTRO, ZOOM_INIT);
L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
    attribution:"CartoDB", maxZoom:19
}}).addTo(map);

function filtrarDatos() {{
    return DATOS.filter(d => {{
        if (filtros.transaccion !== "todos" && d.transaccion !== filtros.transaccion) return false;
        if (filtros.tipo !== "todos" && d.tipo !== filtros.tipo) return false;
        if (filtros.anio !== 0 && d.anio !== filtros.anio) return false;
        if (filtros.ciudad && filtros.ciudad !== "todos" && d.ciudad !== filtros.ciudad) return false;
        return true;
    }});
}}

function fmt(n) {{
    if (!n || n === 0) return "—";
    if (n >= 1000000) return (n/1000000).toFixed(1) + "M BOB";
    if (n >= 1000)    return (n/1000).toFixed(0) + "K BOB";
    return n.toFixed(0) + " BOB";
}}

function actualizarStats(datos) {{
    if (!datos.length) {{
        ["st-total","st-prom","st-min","st-max","st-m2","st-ops",
         "leg-min","leg-max"].forEach(id =>
            document.getElementById(id).textContent = "—"
        );
        return;
    }}
    const precios = datos.map(d => d.precio).filter(p => p > 0);
    const m2s     = datos.map(d => d.precio_m2).filter(p => p > 0);
    const prom    = precios.reduce((a,b)=>a+b,0) / precios.length;
    const mn      = Math.min(...precios);
    const mx      = Math.max(...precios);
    const m2p     = m2s.length ? m2s.reduce((a,b)=>a+b,0)/m2s.length : 0;

    document.getElementById("st-total").textContent = datos.length.toLocaleString();
    document.getElementById("st-prom").textContent  = fmt(prom);
    document.getElementById("st-min").textContent   = fmt(mn);
    document.getElementById("st-max").textContent   = fmt(mx);
    document.getElementById("st-m2").textContent    = m2p > 0 ? fmt(m2p)+"/m²" : "—";
    document.getElementById("st-ops").textContent   = datos.length.toLocaleString() + " ops";
    document.getElementById("leg-min").textContent  = fmt(mn);
    document.getElementById("leg-max").textContent  = fmt(mx);
}}

function limpiarCapas() {{
    if (layerHeat)     {{ map.removeLayer(layerHeat);     layerHeat = null; }}
    if (layerPuntos)   {{ map.removeLayer(layerPuntos);   layerPuntos = null; }}
    if (layerClusters) {{ map.removeLayer(layerClusters); layerClusters = null; }}
    if (layerVoronoi)  {{ map.removeLayer(layerVoronoi);  layerVoronoi = null; }}
}}

function renderHeatmap(datos) {{
    const precios = datos.map(d => d.precio).filter(p => p > 0);
    if (!precios.length) return;
    const pMin = Math.min(...precios);
    const pMax = Math.max(...precios);

    let puntos;
    if (modoHeatmap === "precio") {{
        puntos = datos
            .filter(d => d.precio > 0)
            .map(d => [d.lat, d.lng, (d.precio - pMin) / (pMax - pMin || 1)]);
        document.getElementById("leyenda-titulo").textContent = "Escala de precios";
        document.getElementById("leyenda-grad").style.background =
            "linear-gradient(to right,#00FF00,#FFFF00,#FF0000)";
        document.getElementById("leg-mid").textContent = "precio promedio";
    }} else {{
        // Densidad — todos los puntos con intensidad 1
        puntos = datos.map(d => [d.lat, d.lng, 1]);
        document.getElementById("leyenda-titulo").textContent = "Actividad / Operaciones";
        document.getElementById("leyenda-grad").style.background =
            "linear-gradient(to right,#0000FF,#00FFFF,#FF6600)";
        document.getElementById("leg-mid").textContent = "densidad de ops";
        document.getElementById("leg-min").textContent = "Poca actividad";
        document.getElementById("leg-max").textContent = "Mucha actividad";
    }}

    const gradient = modoHeatmap === "precio"
        ? {{ 0.0:"#00FF00", 0.4:"#FFFF00", 0.7:"#FFA500", 1.0:"#FF0000" }}
        : {{ 0.0:"#0000FF", 0.3:"#00FFFF", 0.6:"#FFAA00", 1.0:"#FF4400" }};

    layerHeat = L.heatLayer(puntos, {{
        radius: 25, blur: 20, maxZoom: 15, gradient
    }}).addTo(map);
}}

function renderCapa(datos) {{
    const precios = datos.map(d => d.precio).filter(p => p > 0);
    if (!precios.length) return;
    const pMin = Math.min(...precios);
    const pMax = Math.max(...precios);

    function getColor(precio) {{
        const r = (precio - pMin) / (pMax - pMin || 1);
        let rv, gv;
        if (r < 0.5) {{ rv = Math.round(255*r*2); gv = 255; }}
        else         {{ rv = 255; gv = Math.round(255*(1-r)*2); }}
        return `rgb(${{rv}},${{gv}},0)`;
    }}

    if (modoCapa === "puntos") {{
        const grupo = L.layerGroup();
        datos.forEach(d => {{
            if (!d.precio) return;
            const color  = getColor(d.precio);
            const marker = L.circleMarker([d.lat, d.lng], {{
                radius:6, color, fillColor:color, fillOpacity:0.85, weight:1
            }});
            marker.bindTooltip(`
                <div class="tt-titulo">${{d.tipo}} · ${{d.transaccion}}</div>
                <div class="tt-fila"><span class="tt-lbl">MLSID</span><span class="tt-val">${{d.mlsid}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Ciudad</span><span class="tt-val">${{d.ciudad}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Precio</span><span class="tt-val">${{fmt(d.precio)}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Precio/m²</span><span class="tt-val">${{d.precio_m2>0?fmt(d.precio_m2)+"/m²":"—"}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Construidos</span><span class="tt-val">${{d.m2_construidos>0?d.m2_construidos+" m²":"—"}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Terreno</span><span class="tt-val">${{d.m2_terreno>0?d.m2_terreno+" m²":"—"}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Dorm/Baños</span><span class="tt-val">${{d.dormitorios}}/${{d.banos}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Zona</span><span class="tt-val">Cluster ${{d.cluster}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Año</span><span class="tt-val">${{d.anio}}</span></div>
            `, {{ sticky:true }});
            grupo.addLayer(marker);
        }});
        layerPuntos = grupo;
        map.addLayer(layerPuntos);

    }} else if (modoCapa === "clusters") {{
        const stats = {{}};
        datos.forEach(d => {{
            const key = d.ciudad + "_" + d.cluster;
            if (!stats[key]) stats[key] = {{ suma:0, count:0, suma_m2:0, count_m2:0, ciudad:d.ciudad, cluster:d.cluster }};
            if (d.precio > 0)    {{ stats[key].suma += d.precio; stats[key].count++; }}
            if (d.precio_m2 > 0) {{ stats[key].suma_m2 += d.precio_m2; stats[key].count_m2++; }}
        }});

        const promedios = Object.values(stats).filter(s => s.count > 0).map(s => s.suma/s.count);
        const pcMin = Math.min(...promedios);
        const pcMax = Math.max(...promedios);

        const grupo = L.layerGroup();
        CENTROIDES.forEach(c => {{
            const key   = c.ciudad + "_" + c.cluster_id;
            const s     = stats[key];
            if (!s || s.count === 0) return;
            const prom  = s.suma / s.count;
            const m2p   = s.count_m2 > 0 ? s.suma_m2/s.count_m2 : 0;
            const ratio = (prom - pcMin) / (pcMax - pcMin || 1);
            let rv, gv;
            if (ratio < 0.5) {{ rv = Math.round(255*ratio*2); gv = 255; }}
            else             {{ rv = 255; gv = Math.round(255*(1-ratio)*2); }}
            const color = `rgb(${{rv}},${{gv}},0)`;

            L.circle([c.centroide_lat, c.centroide_lng], {{
                radius:800, color, fillColor:color, fillOpacity:0.3, weight:2
            }}).addTo(grupo);

            const icon = L.divIcon({{
                html:`<div style="background:${{color}};color:white;font-weight:bold;
                    font-size:12px;width:30px;height:30px;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3)">${{c.cluster_id}}</div>`,
                iconSize:[30,30], iconAnchor:[15,15], className:""
            }});
            L.marker([c.centroide_lat, c.centroide_lng], {{icon}})
             .bindTooltip(`
                <div class="tt-titulo">Zona ${{c.cluster_id}} — ${{c.ciudad}}</div>
                <div class="tt-fila"><span class="tt-lbl">Precio promedio</span><span class="tt-val">${{fmt(prom)}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Precio/m² prom.</span><span class="tt-val">${{m2p>0?fmt(m2p)+"/m²":"—"}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Operaciones</span><span class="tt-val">${{s.count}}</span></div>
             `, {{sticky:true}})
             .addTo(grupo);
        }});
        layerClusters = grupo;
        map.addLayer(layerClusters);

    }} else if (modoCapa === "voronoi") {{
        // Calcular stats por cluster para colorear Voronoi
        const stats = {{}};
        datos.forEach(d => {{
            const key = d.ciudad + "_" + d.cluster;
            if (!stats[key]) stats[key] = {{ suma:0, count:0, ciudad:d.ciudad, cluster:d.cluster }};
            if (d.precio > 0) {{ stats[key].suma += d.precio; stats[key].count++; }}
        }});
        const promedios = Object.values(stats).filter(s=>s.count>0).map(s=>s.suma/s.count);
        const pvMin = Math.min(...promedios);
        const pvMax = Math.max(...promedios);

        const grupo = L.layerGroup();
        VORONOI.forEach(pol => {{
            const key  = pol.ciudad + "_" + pol.cluster_id;
            const s    = stats[key];
            if (!s || s.count === 0) return;
            const prom  = s.suma / s.count;
            const ratio = (prom - pvMin) / (pvMax - pvMin || 1);
            let rv, gv;
            if (ratio < 0.5) {{ rv = Math.round(255*ratio*2); gv = 255; }}
            else             {{ rv = 255; gv = Math.round(255*(1-ratio)*2); }}
            const color = `rgb(${{rv}},${{gv}},0)`;

            // Convertir [lat,lng] a formato Leaflet [[lat,lng],...]
            const coords = pol.coords.map(c => [c[0], c[1]]);
            L.polygon(coords, {{
                color:       color,
                fillColor:   color,
                fillOpacity: 0.35,
                weight:      2,
            }}).bindTooltip(`
                <div class="tt-titulo">Zona ${{pol.cluster_id}} — ${{pol.ciudad}}</div>
                <div class="tt-fila"><span class="tt-lbl">Precio promedio</span><span class="tt-val">${{fmt(prom)}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Operaciones</span><span class="tt-val">${{s.count}}</span></div>
            `, {{sticky:true}})
            .addTo(grupo);
        }});
        layerVoronoi = grupo;
        map.addLayer(layerVoronoi);
    }}
}}

function renderizar() {{
    limpiarCapas();
    const datos = filtrarDatos();
    actualizarStats(datos);
    if (!datos.length) return;
    renderHeatmap(datos);
    if (modoCapa !== "ninguno") renderCapa(datos);
}}

function setHeatmap(modo) {{
    modoHeatmap = modo;
    ["precio","densidad"].forEach(id =>
        document.getElementById("tab-"+id).classList.toggle("activo", id === modo)
    );
    renderizar();
}}

function setCapa(capa) {{
    modoCapa = capa;
    ["ninguno","puntos","clusters","voronoi"].forEach(id =>
        document.getElementById("btn-"+id).classList.toggle("activo", id === capa)
    );
    renderizar();
}}

function setupFiltros(grupoId, campo) {{
    const el = document.getElementById(grupoId);
    if (!el) return;
    el.addEventListener("click", e => {{
        if (!e.target.classList.contains("btn-filtro")) return;
        el.querySelectorAll(".btn-filtro").forEach(b => b.classList.remove("activo"));
        e.target.classList.add("activo");
        const val = e.target.dataset.val;
        filtros[campo] = campo === "anio" ? parseInt(val) : val;
        renderizar();
    }});
}}

setupFiltros("filtro-transaccion", "transaccion");
setupFiltros("filtro-tipo",        "tipo");
setupFiltros("filtro-anio",        "anio");
{setup_ciudad_js}

renderizar();
</script>
</body>
</html>"""

    os.makedirs("data", exist_ok=True)
    nombre = "mapa_precios_bolivia.html" if modo_bolivia else "mapa_precios.html"
    ruta   = f"data/{nombre}"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"\n✓ Mapa guardado en {ruta}")
    webbrowser.open(f"file:///{os.path.abspath(ruta)}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    log.info("="*55)
    log.info("INTRAMAX — Mapa de Calor de Precios v2")
    log.info("="*55)

    # Sin argumento → Bolivia completo
    # Con argumento → ciudad específica
    # Ejemplos:
    #   python visualizar_heatmap.py
    #   python visualizar_heatmap.py "Santa Cruz de la Sierra"
    #   python visualizar_heatmap.py Cochabamba

    if len(sys.argv) > 1:
        ciudad = " ".join(sys.argv[1:])
        log.info(f"  Ciudad: {ciudad}")
        generar_mapa(ciudad)
    else:
        log.info("  Modo: Bolivia completo")
        generar_mapa(None)