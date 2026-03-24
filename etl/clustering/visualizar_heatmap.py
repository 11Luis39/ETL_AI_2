import folium
from folium.plugins import HeatMap, Fullscreen, MiniMap
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import webbrowser
import logging
import sys
from sqlalchemy import text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Colores por cluster — igual que visualizar_clusters.py
# ------------------------------------------------------------
COLORES_HEX = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FFE119", "#FABEBE",
]

COLORES_ICONOS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue", "lightred",
]

def get_pg_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USERNAME')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
    )
    return create_engine(url)

# ------------------------------------------------------------
# Cargar datos desde PostgreSQL
# ------------------------------------------------------------
def cargar_datos(ciudad: str = "Santa Cruz de la Sierra") -> pd.DataFrame:
    engine = get_pg_engine()

    query = text("""
        SELECT
            id_propiedad,
            mlsid,
            tipo_propiedad,
            tipo_transaccion,
            cluster_zona,
            ciudad,
            latitude,
            longitude,
            precio_venta,
            precio_alquiler_mes,
            precio_m2,
            m2_construidos,
            m2_terreno,
            dormitorios,
            banos,
            fecha_venta,
            EXTRACT(YEAR FROM fecha_venta)  AS anio_venta,
            EXTRACT(MONTH FROM fecha_venta) AS mes_venta
        FROM property_analytics
        WHERE ciudad = :ciudad
          AND latitude  IS NOT NULL
          AND longitude IS NOT NULL
          AND (precio_venta IS NOT NULL OR precio_alquiler_mes IS NOT NULL)
        ORDER BY fecha_venta DESC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ciudad": ciudad})

    log.info(f"  → {len(df)} propiedades cargadas para {ciudad}")
    return df

def cargar_centroides(ciudad: str) -> pd.DataFrame:
    engine = get_pg_engine()
    # ENVOLVER EN text() ES OBLIGATORIO
    query = text("""
        SELECT cluster_id, centroide_lat, centroide_lng, total_propiedades 
        FROM zona_clusters 
        WHERE ciudad = :ciudad 
        ORDER BY cluster_id
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(
            query, 
            conn, 
            params={"ciudad": ciudad}
        )
    return df

# ------------------------------------------------------------
# Calcular precio por cluster para choropleth
# ------------------------------------------------------------
def calcular_stats_cluster(df: pd.DataFrame) -> pd.DataFrame:
    col_precio = "precio_venta" if "precio_venta" in df.columns else "precio_alquiler_mes"

    stats = df.groupby("cluster_zona").agg(
        precio_promedio  = (col_precio, "mean"),
        precio_mediana   = (col_precio, "median"),
        precio_m2_prom   = ("precio_m2", "mean"),
        total            = (col_precio, "count"),
        precio_min       = (col_precio, "min"),
        precio_max       = (col_precio, "max"),
    ).reset_index()

    return stats

# ------------------------------------------------------------
# Generar color por precio (gradiente verde → amarillo → rojo)
# ------------------------------------------------------------
def precio_a_color(precio: float, precio_min: float, precio_max: float) -> str:
    if precio_max == precio_min:
        return "#FFFF00"
    ratio = (precio - precio_min) / (precio_max - precio_min)
    ratio = max(0, min(1, ratio))

    if ratio < 0.5:
        r = int(255 * ratio * 2)
        g = 255
    else:
        r = 255
        g = int(255 * (1 - ratio) * 2)
    b = 0
    return f"#{r:02X}{g:02X}{b:02X}"

# ------------------------------------------------------------
# Generar mapa completo con filtros HTML
# ------------------------------------------------------------
def generar_mapa(ciudad: str = "Santa Cruz de la Sierra"):
    log.info(f"Cargando datos para {ciudad}...")
    df_all       = cargar_datos(ciudad)
    df_centroides = cargar_centroides(ciudad)

    if df_all.empty:
        log.error("No hay datos. Corré el ETL primero.")
        return

    # Valores únicos para los filtros
    tipos_prop   = sorted(df_all["tipo_propiedad"].dropna().unique().tolist())
    tipos_trans  = sorted(df_all["tipo_transaccion"].dropna().unique().tolist())
    anios        = sorted(df_all["anio_venta"].dropna().unique().astype(int).tolist(), reverse=True)

    # Centro del mapa
    centro_lat = df_all["latitude"].mean()
    centro_lng = df_all["longitude"].mean()

    # Preparar datos JSON para JavaScript
    import json

    registros = []
    for _, row in df_all.iterrows():
        # Lógica unificada: si no hay precio_venta, busca precio_alquiler_mes
        precio_val = row.get("precio_venta")
        if pd.isna(precio_val) or precio_val == 0:
            precio_val = row.get("precio_alquiler_mes")
        
        # Solo agregamos si logramos obtener algún precio válido
        if pd.notnull(precio_val) and precio_val > 0:
            registros.append({
                "lat":         float(row["latitude"]),
                "lng":         float(row["longitude"]),
                "mlsid":       str(row["mlsid"]) if pd.notnull(row["mlsid"]) else "N/A",
                "precio":      float(precio_val),
                "precio_m2":   float(row["precio_m2"]) if pd.notnull(row["precio_m2"]) else 0,
                "tipo":        str(row["tipo_propiedad"] or ""),
                "transaccion": str(row["tipo_transaccion"] or ""),
                "cluster":     int(row["cluster_zona"]) if pd.notnull(row["cluster_zona"]) else -1,
                "anio":        int(row["anio_venta"]) if pd.notnull(row["anio_venta"]) else 0,
                "m2_construidos": float(row["m2_construidos"]) if pd.notnull(row["m2_construidos"]) else 0,
                "m2_terreno": float(row["m2_terreno"]) if pd.notnull(row["m2_terreno"]) else 0,
                "dormitorios": int(row["dormitorios"]) if pd.notnull(row["dormitorios"]) else 0,
                "banos":       int(row["banos"]) if pd.notnull(row["banos"]) else 0,
            })

    centroides_json = []
    for _, row in df_centroides.iterrows():
        centroides_json.append({
            "cluster_id": int(row["cluster_id"]),
            "lat":        float(row["centroide_lat"]),
            "lng":        float(row["centroide_lng"]),
            "total":      int(row["total_propiedades"]),
        })

    colores_js = json.dumps(COLORES_HEX)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>INTRAMAX · Mapa de Precios — {ciudad}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; }}

        #app {{ display: flex; height: 100vh; }}

        /* Panel lateral */
        #panel {{
            width: 300px;
            min-width: 300px;
            background: white;
            box-shadow: 2px 0 12px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            z-index: 1000;
            overflow-y: auto;
        }}

        #panel-header {{
            background: #1e3a5f;
            color: white;
            padding: 16px;
        }}
        #panel-header h2 {{ font-size: 15px; font-weight: 700; }}
        #panel-header p  {{ font-size: 11px; opacity: 0.8; margin-top: 4px; }}

        .filtro-grupo {{
            padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .filtro-grupo label {{
            font-size: 11px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: block;
            margin-bottom: 6px;
        }}

        .btn-grupo {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}
        .btn-filtro {{
            padding: 4px 10px;
            border: 1px solid #ddd;
            border-radius: 20px;
            background: white;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s;
            color: #333;
        }}
        .btn-filtro:hover {{ border-color: #1e3a5f; color: #1e3a5f; }}
        .btn-filtro.activo {{
            background: #1e3a5f;
            color: white;
            border-color: #1e3a5f;
        }}

        select {{
            width: 100%;
            padding: 6px 8px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 12px;
            color: #333;
            background: white;
        }}

        /* Vista */
        .vista-btn {{
            display: flex;
            gap: 6px;
        }}
        .vista-btn button {{
            flex: 1;
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .vista-btn button.activo {{
            background: #1e3a5f;
            color: white;
            border-color: #1e3a5f;
        }}

        /* Stats */
        #stats-panel {{
            padding: 12px 16px;
            border-bottom: 1px solid #f0f0f0;
        }}
        #stats-panel h3 {{ font-size: 12px; font-weight: 600; color: #666; margin-bottom: 8px; text-transform: uppercase; }}

        .stat-item {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 4px;
        }}
        .stat-item .lbl {{ color: #888; }}
        .stat-item .val {{ font-weight: 600; color: #1e3a5f; }}

        /* Leyenda */
        #leyenda {{
            padding: 12px 16px;
        }}
        #leyenda h3 {{ font-size: 12px; font-weight: 600; color: #666; margin-bottom: 8px; text-transform: uppercase; }}
        .leyenda-gradiente {{
            height: 12px;
            border-radius: 6px;
            background: linear-gradient(to right, #00FF00, #FFFF00, #FF0000);
            margin-bottom: 4px;
        }}
        .leyenda-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #888;
        }}

        /* Mapa */
        #map {{ flex: 1; }}

        /* Tooltip personalizado */
        .leaflet-tooltip {{
            background: white;
            border: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 12px;
        }}
        .tooltip-titulo {{ font-weight: 700; color: #1e3a5f; margin-bottom: 4px; font-size: 13px; }}
        .tooltip-fila {{ display: flex; justify-content: space-between; gap: 16px; margin-bottom: 2px; }}
        .tooltip-lbl {{ color: #888; }}
        .tooltip-val {{ font-weight: 600; }}
    </style>
</head>
<body>
<div id="app">

    <!-- Panel de filtros -->
    <div id="panel">
        <div id="panel-header">
            <h2>🏠 Mapa de Precios</h2>
            <p>INTRAMAX · {ciudad}</p>
        </div>

        <!-- Tipo de transacción -->
        <div class="filtro-grupo">
            <label>Tipo de transacción</label>
            <div class="btn-grupo" id="filtro-transaccion">
                <button class="btn-filtro activo" data-val="todos">Todos</button>
                {''.join(f'<button class="btn-filtro" data-val="{t}">{t}</button>' for t in tipos_trans)}
            </div>
        </div>

        <!-- Tipo de propiedad -->
        <div class="filtro-grupo">
            <label>Tipo de propiedad</label>
            <div class="btn-grupo" id="filtro-tipo">
                <button class="btn-filtro activo" data-val="todos">Todos</button>
                {''.join(f'<button class="btn-filtro" data-val="{t}">{t}</button>' for t in tipos_prop)}
            </div>
        </div>

        <!-- Año -->
        <div class="filtro-grupo">
            <label>Año de venta</label>
            <div class="btn-grupo" id="filtro-anio">
                <button class="btn-filtro activo" data-val="0">Todos</button>
                {''.join(f'<button class="btn-filtro" data-val="{a}">{a}</button>' for a in anios)}
            </div>
        </div>

        <!-- Vista -->
        <div class="filtro-grupo">
            <label>Vista del mapa</label>
            <div class="vista-btn">
                <button id="btn-heatmap" class="activo" onclick="setVista('heatmap')">🌡️ Calor</button>
                <button id="btn-puntos"  onclick="setVista('puntos')">📍 Puntos</button>
                <button id="btn-clusters" onclick="setVista('clusters')">⭕ Zonas</button>
            </div>
        </div>

        <!-- Estadísticas -->
        <div id="stats-panel">
            <h3>Resumen filtrado</h3>
            <div class="stat-item"><span class="lbl">Propiedades</span><span class="val" id="st-total">—</span></div>
            <div class="stat-item"><span class="lbl">Precio promedio</span><span class="val" id="st-prom">—</span></div>
            <div class="stat-item"><span class="lbl">Precio mínimo</span><span class="val" id="st-min">—</span></div>
            <div class="stat-item"><span class="lbl">Precio máximo</span><span class="val" id="st-max">—</span></div>
            <div class="stat-item"><span class="lbl">Precio/m² prom.</span><span class="val" id="st-m2">—</span></div>
        </div>

        <!-- Leyenda -->
        <div id="leyenda">
            <h3>Escala de precios</h3>
            <div class="leyenda-gradiente"></div>
            <div class="leyenda-labels">
                <span id="leg-min">—</span>
                <span>Precio promedio</span>
                <span id="leg-max">—</span>
            </div>
        </div>
    </div>

    <!-- Mapa -->
    <div id="map"></div>
</div>

<script>
// ── Datos desde Python ──────────────────────────────────────
const DATOS       = {json.dumps(registros)};
const CENTROIDES  = {json.dumps(centroides_json)};
const COLORES     = {colores_js};
const CENTRO      = [{centro_lat}, {centro_lng}];

// ── Estado de filtros ───────────────────────────────────────
let filtros = {{ transaccion: "todos", tipo: "todos", anio: 0 }};
let vista   = "heatmap";

// ── Capas del mapa ──────────────────────────────────────────
let heatLayer    = null;
let puntosLayer  = null;
let clustersLayer = null;

// ── Inicializar mapa ────────────────────────────────────────
const map = L.map("map").setView(CENTRO, 12);

L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
    attribution: "CartoDB",
    maxZoom: 19,
}}).addTo(map);

// ── Filtrar datos ───────────────────────────────────────────
function filtrarDatos() {{
    return DATOS.filter(d => {{
        if (filtros.transaccion !== "todos" && d.transaccion !== filtros.transaccion) return false;
        if (filtros.tipo !== "todos" && d.tipo !== filtros.tipo) return false;
        if (filtros.anio !== 0 && d.anio !== filtros.anio) return false;
        return true;
    }});
}}

// ── Formatear precio ────────────────────────────────────────
function fmt(n) {{
    if (!n || n === 0) return "—";
    if (n >= 1000000) return (n/1000000).toFixed(1) + "M BOB";
    if (n >= 1000)    return (n/1000).toFixed(0) + "K BOB";
    return n.toFixed(0) + " BOB";
}}

// ── Actualizar estadísticas ─────────────────────────────────
function actualizarStats(datos) {{
    if (datos.length === 0) {{
        document.getElementById("st-total").textContent = "0";
        ["st-prom","st-min","st-max","st-m2","leg-min","leg-max"].forEach(id =>
            document.getElementById(id).textContent = "—"
        );
        return;
    }}
    const precios = datos.map(d => d.precio).filter(p => p > 0);
    const m2s     = datos.map(d => d.precio_m2).filter(p => p > 0);
    const prom    = precios.reduce((a,b) => a+b, 0) / precios.length;
    const mn      = Math.min(...precios);
    const mx      = Math.max(...precios);
    const m2prom  = m2s.length > 0 ? m2s.reduce((a,b) => a+b, 0) / m2s.length : 0;

    document.getElementById("st-total").textContent = datos.length.toLocaleString();
    document.getElementById("st-prom").textContent  = fmt(prom);
    document.getElementById("st-min").textContent   = fmt(mn);
    document.getElementById("st-max").textContent   = fmt(mx);
    document.getElementById("st-m2").textContent    = m2prom > 0 ? fmt(m2prom) + "/m²" : "—";
    document.getElementById("leg-min").textContent  = fmt(mn);
    document.getElementById("leg-max").textContent  = fmt(mx);
}}

// ── Renderizar capas ────────────────────────────────────────
function renderizar() {{
    const datos = filtrarDatos();
    actualizarStats(datos);

    // Limpiar capas anteriores
    if (heatLayer)     {{ map.removeLayer(heatLayer);     heatLayer = null; }}
    if (puntosLayer)   {{ map.removeLayer(puntosLayer);   puntosLayer = null; }}
    if (clustersLayer) {{ map.removeLayer(clustersLayer); clustersLayer = null; }}

    if (datos.length === 0) return;

    const precios = datos.map(d => d.precio).filter(p => p > 0);
    const pMin    = Math.min(...precios);
    const pMax    = Math.max(...precios);

    if (vista === "heatmap") {{
        // Normalizar intensidad por precio
        const puntos = datos
            .filter(d => d.precio > 0)
            .map(d => [d.lat, d.lng, (d.precio - pMin) / (pMax - pMin || 1)]);

        heatLayer = L.heatLayer(puntos, {{
            radius:  25,
            blur:    20,
            maxZoom: 15,
            gradient: {{ 0.0: "#00FF00", 0.4: "#FFFF00", 0.7: "#FFA500", 1.0: "#FF0000" }},
        }}).addTo(map);

    }} else if (vista === "puntos") {{
        const grupo = L.layerGroup();
        datos.forEach(d => {{
            if (!d.precio || d.precio === 0) return;
            const ratio = (d.precio - pMin) / (pMax - pMin || 1);
            let r, g;
            if (ratio < 0.5) {{ r = Math.round(255 * ratio * 2); g = 255; }}
            else             {{ r = 255; g = Math.round(255 * (1 - ratio) * 2); }}
            const color = `rgb(${{r}},${{g}},0)`;

            const circle = L.circleMarker([d.lat, d.lng], {{
                radius:      6,
                color:       color,
                fillColor:   color,
                fillOpacity: 0.85,
                weight:      1,
            }});

            const tooltipHtml = `
                <div class="tooltip-titulo">${{d.tipo}} · ${{d.transaccion}}</div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">MLSID</span>
                    <span class="tooltip-val">${{(d.mlsid)}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Precio</span>
                    <span class="tooltip-val">${{fmt(d.precio)}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Precio/m²</span>
                    <span class="tooltip-val">${{d.precio_m2 > 0 ? fmt(d.precio_m2) + "/m²" : "—"}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">m² construidos</span>
                    <span class="tooltip-val">${{d.m2_construidos > 0 ? d.m2_construidos + " m²" : "—"}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">m² terreno</span>
                    <span class="tooltip-val">${{d.m2_terreno > 0 ? d.m2_terreno + " m²" : "—"}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Dorm / Baños</span>
                    <span class="tooltip-val">${{d.dormitorios}} / ${{d.banos}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Zona</span>
                    <span class="tooltip-val">Cluster ${{d.cluster}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Año</span>
                    <span class="tooltip-val">${{d.anio}}</span>
                </div>
            `;
            circle.bindTooltip(tooltipHtml, {{ sticky: true }});
            grupo.addLayer(circle);
        }});
        puntosLayer = grupo;
        map.addLayer(puntosLayer);

    }} else if (vista === "clusters") {{
        // Calcular precio promedio por cluster con datos filtrados
        const statsPorCluster = {{}};
        datos.forEach(d => {{
            if (!statsPorCluster[d.cluster]) {{
                statsPorCluster[d.cluster] = {{ suma: 0, count: 0, suma_m2: 0, count_m2: 0 }};
            }}
            if (d.precio > 0) {{
                statsPorCluster[d.cluster].suma  += d.precio;
                statsPorCluster[d.cluster].count += 1;
            }}
            if (d.precio_m2 > 0) {{
                statsPorCluster[d.cluster].suma_m2  += d.precio_m2;
                statsPorCluster[d.cluster].count_m2 += 1;
            }}
        }});

        const promedios = Object.entries(statsPorCluster)
            .filter(([_, s]) => s.count > 0)
            .map(([cid, s]) => s.suma / s.count);

        const pMinC = Math.min(...promedios);
        const pMaxC = Math.max(...promedios);

        const grupo = L.layerGroup();

        CENTROIDES.forEach(c => {{
            const stats = statsPorCluster[c.cluster_id];
            if (!stats || stats.count === 0) return;

            const promedio = stats.suma / stats.count;
            const m2prom   = stats.count_m2 > 0 ? stats.suma_m2 / stats.count_m2 : 0;
            const ratio    = (promedio - pMinC) / (pMaxC - pMinC || 1);

            let r, g;
            if (ratio < 0.5) {{ r = Math.round(255 * ratio * 2); g = 255; }}
            else             {{ r = 255; g = Math.round(255 * (1 - ratio) * 2); }}
            const color = `rgb(${{r}},${{g}},0)`;

            // Círculo de zona
            L.circle([c.lat, c.lng], {{
                radius:      800,
                color:       color,
                fillColor:   color,
                fillOpacity: 0.35,
                weight:      2,
            }}).addTo(grupo);

            // Marcador con número de cluster
            const icon = L.divIcon({{
                html: `<div style="background:${{color}};color:white;font-weight:bold;
                               font-size:13px;width:32px;height:32px;border-radius:50%;
                               display:flex;align-items:center;justify-content:center;
                               border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3)">
                           ${{c.cluster_id}}
                       </div>`,
                iconSize:   [32, 32],
                iconAnchor: [16, 16],
                className:  "",
            }});

            const tooltipHtml = `
                <div class="tooltip-titulo">Zona ${{c.cluster_id}}</div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Precio promedio</span>
                    <span class="tooltip-val">${{fmt(promedio)}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Precio/m² prom.</span>
                    <span class="tooltip-val">${{m2prom > 0 ? fmt(m2prom) + "/m²" : "—"}}</span>
                </div>
                <div class="tooltip-fila">
                    <span class="tooltip-lbl">Propiedades</span>
                    <span class="tooltip-val">${{stats.count}}</span>
                </div>
            `;

            L.marker([c.lat, c.lng], {{ icon }})
             .bindTooltip(tooltipHtml, {{ sticky: true }})
             .addTo(grupo);
        }});

        clustersLayer = grupo;
        map.addLayer(clustersLayer);
    }}
}}

// ── Cambiar vista ───────────────────────────────────────────
function setVista(v) {{
    vista = v;
    ["heatmap","puntos","clusters"].forEach(id => {{
        document.getElementById("btn-" + id).classList.toggle("activo", id === v);
    }});
    renderizar();
}}

// ── Eventos de filtros ──────────────────────────────────────
function setupFiltros(grupoId, campo) {{
    document.getElementById(grupoId).addEventListener("click", e => {{
        if (!e.target.classList.contains("btn-filtro")) return;
        document.querySelectorAll(`#${{grupoId}} .btn-filtro`).forEach(b => b.classList.remove("activo"));
        e.target.classList.add("activo");
        const val = e.target.dataset.val;
        filtros[campo] = campo === "anio" ? parseInt(val) : val;
        renderizar();
    }});
}}

setupFiltros("filtro-transaccion", "transaccion");
setupFiltros("filtro-tipo",        "tipo");
setupFiltros("filtro-anio",        "anio");

// ── Render inicial ──────────────────────────────────────────
renderizar();
</script>
</body>
</html>"""

    os.makedirs("data", exist_ok=True)
    ruta = "data/mapa_precios.html"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"\n✓ Mapa guardado en {ruta}")
    webbrowser.open(f"file:///{os.path.abspath(ruta)}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    log.info("="*55)
    log.info("INTRAMAX — Mapa de Calor de Precios")
    log.info("="*55)

    import sys
    ciudad = sys.argv[1] if len(sys.argv) > 1 else "Santa Cruz de la Sierra"
    generar_mapa(ciudad)