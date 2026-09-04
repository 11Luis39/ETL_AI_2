import json
import logging
import os
import sys
import webbrowser

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sqlalchemy import text

from etl.config import get_pg_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

COLORES_HEX = [
    "#E6194B",
    "#3CB44B",
    "#4363D8",
    "#F58231",
    "#911EB4",
    "#42D4F4",
    "#F032E6",
    "#BFEF45",
    "#FFE119",
    "#FABEBE",
]


# ------------------------------------------------------------
# Cargar datos
# ------------------------------------------------------------
def cargar_datos(ciudad: str = None) -> pd.DataFrame:
    engine = get_pg_engine()
    if ciudad:
        query = text("""
            SELECT id_propiedad, mlsid, tipo_propiedad, tipo_transaccion,
                   cluster_zona, ciudad, latitude, longitude,
                   precio_publicacion, precio_venta, precio_alquiler_mes, precio_m2,
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
                   precio_publicacion, precio_venta, precio_alquiler_mes, precio_m2,
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

    log.info(
        f"  -> {len(df)} propiedades cargadas{' para ' + ciudad if ciudad else ' (Bolivia completo)'}"
    )
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
def calcular_zonas_organicas(
    df_propiedades: pd.DataFrame,
    centroides_df: pd.DataFrame,
) -> list:
    """
    Crea polígonos que rodean las propiedades reales de cada cluster
    usando Convex Hull.

    Ignora clusters que no tengan suficientes coordenadas únicas
    o cuyos puntos sean colineales.
    """
    poligonos = []

    for _, centroide in centroides_df.iterrows():
        c_id = centroide["cluster_id"]
        ciudad = centroide["ciudad"]

        puntos_cluster = df_propiedades[
            (df_propiedades["cluster_zona"] == c_id) & (df_propiedades["ciudad"] == ciudad)
        ][["latitude", "longitude"]].to_numpy(dtype=float)

        # Quitar NaN / infinitos por seguridad
        puntos_cluster = puntos_cluster[np.isfinite(puntos_cluster).all(axis=1)]

        if len(puntos_cluster) < 3:
            log.debug(
                "Cluster %s - %s ignorado: menos de 3 propiedades",
                c_id,
                ciudad,
            )
            continue

        # Varias propiedades pueden compartir exactamente
        # la misma latitud/longitud.
        puntos_unicos = np.unique(puntos_cluster, axis=0)

        if len(puntos_unicos) < 3:
            log.warning(
                "Cluster %s - %s ignorado: %s propiedades pero solo %s coordenadas únicas",
                c_id,
                ciudad,
                len(puntos_cluster),
                len(puntos_unicos),
            )
            continue

        # ConvexHull 2D necesita que los puntos realmente ocupen
        # dos dimensiones. Si están todos sobre una línea, rank será 1.
        puntos_centrados = puntos_unicos - puntos_unicos.mean(axis=0)

        if np.linalg.matrix_rank(puntos_centrados) < 2:
            log.warning(
                "Cluster %s - %s ignorado: coordenadas colineales",
                c_id,
                ciudad,
            )
            continue

        try:
            hull = ConvexHull(puntos_unicos)

            vertices = puntos_unicos[hull.vertices].tolist()

            poligonos.append(
                {
                    "cluster_id": int(c_id),
                    "ciudad": str(ciudad),
                    "coords": vertices,
                }
            )

        except QhullError as exc:
            log.warning(
                "Cluster %s - %s ignorado por ConvexHull: %s",
                c_id,
                ciudad,
                str(exc).splitlines()[0],
            )
            continue

        except (TypeError, ValueError) as exc:
            log.warning(
                "Cluster %s - %s ignorado: %s",
                c_id,
                ciudad,
                exc,
            )
            continue

    log.info(
        "  -> %s polígonos orgánicos generados de %s clusters",
        len(poligonos),
        len(centroides_df),
    )

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

        registros.append(
            {
                "id": str(row["id_propiedad"]),
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "mlsid": str(row["mlsid"]) if pd.notnull(row["mlsid"]) else "N/A",
                "precio": float(precio_val),
                "precio_publicacion": float(row["precio_publicacion"])
                if pd.notnull(row["precio_publicacion"])
                else 0,
                "precio_m2": float(row["precio_m2"]) if pd.notnull(row["precio_m2"]) else 0,
                "tipo": str(row["tipo_propiedad"] or ""),
                "transaccion": str(row["tipo_transaccion"] or ""),
                "ciudad": str(row["ciudad"] or ""),
                "cluster": int(row["cluster_zona"]) if pd.notnull(row["cluster_zona"]) else -1,
                "anio": int(row["anio_venta"]) if pd.notnull(row["anio_venta"]) else 0,
                "m2_construidos": float(row["m2_construidos"])
                if pd.notnull(row["m2_construidos"])
                else 0,
                "m2_terreno": float(row["m2_terreno"]) if pd.notnull(row["m2_terreno"]) else 0,
                "dormitorios": int(row["dormitorios"]) if pd.notnull(row["dormitorios"]) else 0,
                "banos": int(row["banos"]) if pd.notnull(row["banos"]) else 0,
            }
        )
    return registros


# ------------------------------------------------------------
# Generar mapa
# ------------------------------------------------------------
def generar_mapa(ciudad: str = None):
    modo_bolivia = ciudad is None
    titulo_ciudad = "Bolivia — Todas las ciudades" if modo_bolivia else ciudad

    log.info(f"Cargando datos{'...' if modo_bolivia else f' para {ciudad}...'}")
    df_all = cargar_datos(ciudad)
    df_centroides = cargar_centroides(ciudad)
    poligonos_organicos = calcular_zonas_organicas(df_all, df_centroides)

    if df_all.empty:
        log.error("No hay datos. Corré el ETL primero.")
        return

    if modo_bolivia:
        centro_lat, centro_lng = -16.5, -64.5
        zoom_inicial = 6
    else:
        centro_lat = df_all["latitude"].mean()
        centro_lng = df_all["longitude"].mean()
        zoom_inicial = 12

    # Valores únicos para filtros
    tipos_prop = sorted(df_all["tipo_propiedad"].dropna().unique().tolist())
    tipos_trans = sorted(df_all["tipo_transaccion"].dropna().unique().tolist())
    anios = sorted(df_all["anio_venta"].dropna().unique().astype(int).tolist(), reverse=True)
    ciudades = sorted(df_all["ciudad"].dropna().unique().tolist()) if modo_bolivia else []

    registros = preparar_registros(df_all)
    centroides_js = df_centroides.to_dict("records")
    for c in centroides_js:
        c["cluster_id"] = int(c["cluster_id"])
        c["centroide_lat"] = float(c["centroide_lat"])
        c["centroide_lng"] = float(c["centroide_lng"])
        c["total_propiedades"] = int(c["total_propiedades"])

    # Botones de ciudad para Bolivia completo
    botones_ciudad = ""
    if modo_bolivia:
        botones_ciudad = f"""
        <div class="filtro-grupo">
            <label>Ciudad</label>
            <div class="btn-grupo" id="filtro-ciudad">
                <button class="btn-filtro activo" data-val="todos">Todas</button>
                {"".join(f'<button class="btn-filtro" data-val="{c}">{c}</button>' for c in ciudades)}
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
        :root {{
            --ink:#15212a; --navy:#123047; --paper:#f6f8fa;
            --line:#dce4e8; --low:#0f766e; --high:#c2413b;
        }}
        body {{ font-family:Aptos,'Segoe UI',sans-serif; background:var(--paper); color:var(--ink); }}
        #app {{ display:flex; height:100vh; }}

        #panel {{
            width:340px; min-width:340px; background:white;
            box-shadow:2px 0 12px rgba(0,0,0,0.1);
            display:flex; flex-direction:column; z-index:1000; overflow-y:auto;
        }}
        #panel-header {{ background:var(--navy); color:white; padding:18px 16px 16px; }}
        #panel-header h2 {{
            font-family:'Bahnschrift SemiCondensed',Aptos,sans-serif;
            font-size:17px; font-weight:700; letter-spacing:0.2px;
        }}
        #panel-header p  {{ font-size:11px; opacity:0.8; margin-top:4px; }}
        #panel-header .moneda {{
            display:inline-flex; margin-top:10px; padding:3px 8px;
            border:1px solid rgba(255,255,255,.35); border-radius:999px;
            font:700 10px Consolas,monospace; letter-spacing:.08em;
        }}

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

        #extremos-panel {{ padding:2px 10px 10px; }}
        .extremos-encabezado {{
            display:flex; align-items:flex-start; justify-content:space-between;
            gap:12px; margin:0 2px 8px;
        }}
        .extremos-encabezado h3 {{
            font-size:11px; color:#53636e; text-transform:uppercase;
            letter-spacing:.08em;
        }}
        .extremos-encabezado span {{ font-size:9px; color:#7b8992; text-align:right; }}
        .extremo-card {{
            width:100%; display:grid; grid-template-columns:6px 1fr auto;
            gap:10px; align-items:center; padding:10px 10px 10px 8px;
            margin-bottom:7px; border:1px solid var(--line); border-radius:10px;
            background:#fff; text-align:left; cursor:pointer;
            box-shadow:0 3px 12px rgba(18,48,71,.06);
            transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease;
        }}
        .extremo-card:hover {{ transform:translateY(-1px); box-shadow:0 6px 16px rgba(18,48,71,.12); }}
        .extremo-card:focus-visible {{ outline:3px solid rgba(18,48,71,.24); outline-offset:2px; }}
        .extremo-card:disabled {{ opacity:.5; cursor:not-allowed; transform:none; }}
        .extremo-barra {{ align-self:stretch; border-radius:99px; background:var(--low); }}
        .extremo-card.max .extremo-barra {{ background:var(--high); }}
        .extremo-kicker {{
            display:block; font-size:9px; font-weight:800; letter-spacing:.1em;
            text-transform:uppercase; color:var(--low);
        }}
        .extremo-card.max .extremo-kicker {{ color:var(--high); }}
        .extremo-precio {{
            display:block; margin:2px 0 1px;
            font:700 17px 'Bahnschrift SemiCondensed',Aptos,sans-serif; color:var(--ink);
        }}
        .extremo-meta {{ display:block; font-size:10px; color:#687781; line-height:1.35; }}
        .extremo-id {{ font-family:Consolas,monospace; color:#344955; }}
        .extremo-accion {{ font-size:16px; color:#78909c; }}

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

        .price-flag {{
            display:flex; align-items:center; gap:6px; min-width:max-content;
            padding:5px 8px; border-radius:8px; color:white;
            font:700 10px Consolas,monospace; letter-spacing:.03em;
            box-shadow:0 4px 14px rgba(0,0,0,.28); border:2px solid white;
        }}
        .price-flag.low {{ background:var(--low); }}
        .price-flag.high {{ background:var(--high); }}
        .price-popup {{ min-width:220px; }}
        .price-popup .badge {{
            display:inline-block; padding:2px 6px; border-radius:99px;
            background:#eaf2f3; color:var(--navy); font-size:9px;
            font-weight:800; letter-spacing:.08em; text-transform:uppercase;
        }}

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
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ scroll-behavior:auto !important; transition:none !important; }}
        }}
        @media (max-width:780px) {{
            #app {{ flex-direction:column; }}
            #panel {{ width:100%; min-width:100%; max-height:46vh; }}
            #map {{ min-height:54vh; }}
        }}
    </style>
</head>
<body>
<div id="app">
<div id="panel">
    <div id="panel-header">
        <h2>🏠 Mapa de Precios INTRAMAX</h2>
        <p>{titulo_ciudad}</p>
        <span class="moneda">PRECIOS EN USD</span>
    </div>

    <!-- Captaciones extremas según los filtros activos -->
    <div id="extremos-panel">
        <div class="extremos-encabezado">
            <h3>Captaciones extremas</h3>
            <span>Precio de cierre<br>según filtros</span>
        </div>
        <button class="extremo-card min" id="extremo-min" onclick="enfocarExtremo('min')" disabled>
            <span class="extremo-barra"></span>
            <span>
                <span class="extremo-kicker">Menor precio</span>
                <span class="extremo-precio" id="ext-min-precio">—</span>
                <span class="extremo-meta" id="ext-min-meta">Sin datos</span>
            </span>
            <span class="extremo-accion" aria-hidden="true">⌖</span>
        </button>
        <button class="extremo-card max" id="extremo-max" onclick="enfocarExtremo('max')" disabled>
            <span class="extremo-barra"></span>
            <span>
                <span class="extremo-kicker">Mayor precio</span>
                <span class="extremo-precio" id="ext-max-precio">—</span>
                <span class="extremo-meta" id="ext-max-meta">Sin datos</span>
            </span>
            <span class="extremo-accion" aria-hidden="true">⌖</span>
        </button>
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
            {"".join(f'<button class="btn-filtro" data-val="{t}">{t}</button>' for t in tipos_trans)}
        </div>
    </div>

    <!-- Tipo propiedad -->
    <div class="filtro-grupo">
        <label>Tipo de propiedad</label>
        <div class="btn-grupo" id="filtro-tipo">
            <button class="btn-filtro activo" data-val="todos">Todos</button>
            {"".join(f'<button class="btn-filtro" data-val="{t}">{t}</button>' for t in tipos_prop)}
        </div>
    </div>

    <!-- Año -->
    <div class="filtro-grupo">
        <label>Año</label>
        <div class="btn-grupo" id="filtro-anio">
            <button class="btn-filtro activo" data-val="0">Todos</button>
            {"".join(f'<button class="btn-filtro" data-val="{a}">{a}</button>' for a in anios)}
        </div>
    </div>

    <!-- Stats -->
    <div id="stats-panel">
        <h3>Resumen</h3>
        <div class="stat-item"><span class="lbl">Propiedades</span><span class="val" id="st-total">—</span></div>
        <div class="stat-item"><span class="lbl">Cierre promedio</span><span class="val" id="st-prom">—</span></div>
        <div class="stat-item"><span class="lbl">Cierre mínimo</span><span class="val" id="st-min">—</span></div>
        <div class="stat-item"><span class="lbl">Cierre máximo</span><span class="val" id="st-max">—</span></div>
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
let layerExtremos = null;
let extremosActuales = {{ min:null, max:null }};
let marcadoresExtremos = {{ min:null, max:null }};

const map = L.map("map").setView(CENTRO, ZOOM_INIT);
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
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
    return new Intl.NumberFormat("en-US", {{
        style:"currency", currency:"USD", maximumFractionDigits:0
    }}).format(n);
}}

function fmtM2(n) {{
    return n > 0 ? fmt(n) + "/m²" : "—";
}}

function descripcionCaptacion(d) {{
    const identificador = d.mlsid !== "N/A" ? `MLS ${{d.mlsid}}` : `ID ${{d.id}}`;
    return `${{identificador}} · ${{d.tipo}} · ${{d.transaccion}} · ${{d.ciudad}}`;
}}

function actualizarTarjetaExtremo(tipo, captacion) {{
    const boton = document.getElementById(`extremo-${{tipo}}`);
    const precio = document.getElementById(`ext-${{tipo}}-precio`);
    const meta = document.getElementById(`ext-${{tipo}}-meta`);
    boton.disabled = !captacion;
    precio.textContent = captacion ? fmt(captacion.precio) : "—";
    meta.innerHTML = captacion
        ? `<span class="extremo-id">${{esc(captacion.mlsid !== "N/A" ? "MLS " + captacion.mlsid : "ID " + captacion.id)}}</span><br>${{esc(captacion.tipo)}} · ${{esc(captacion.transaccion)}} · ${{esc(captacion.ciudad)}}`
        : "Sin datos para estos filtros";
    boton.setAttribute(
        "aria-label",
        captacion ? `Ver captación de ${{tipo === "min" ? "menor" : "mayor"}} precio: ${{descripcionCaptacion(captacion)}}` : "Sin datos"
    );
}}

function actualizarStats(datos) {{
    if (!datos.length) {{
        ["st-total","st-prom","st-min","st-max","st-m2","st-ops",
         "leg-min","leg-max"].forEach(id =>
            document.getElementById(id).textContent = "—"
        );
        extremosActuales = {{ min:null, max:null }};
        actualizarTarjetaExtremo("min", null);
        actualizarTarjetaExtremo("max", null);
        return;
    }}
    const precios = datos.map(d => d.precio).filter(p => p > 0);
    const m2s     = datos.map(d => d.precio_m2).filter(p => p > 0);
    const prom    = precios.reduce((a,b)=>a+b,0) / precios.length;
    const validos = datos.filter(d => d.precio > 0);
    const minCaptacion = validos.reduce((menor, actual) => actual.precio < menor.precio ? actual : menor);
    const maxCaptacion = validos.reduce((mayor, actual) => actual.precio > mayor.precio ? actual : mayor);
    const mn      = minCaptacion.precio;
    const mx      = maxCaptacion.precio;
    const m2p     = m2s.length ? m2s.reduce((a,b)=>a+b,0)/m2s.length : 0;

    document.getElementById("st-total").textContent = datos.length.toLocaleString();
    document.getElementById("st-prom").textContent  = fmt(prom);
    document.getElementById("st-min").textContent   = fmt(mn);
    document.getElementById("st-max").textContent   = fmt(mx);
    document.getElementById("st-m2").textContent    = fmtM2(m2p);
    document.getElementById("st-ops").textContent   = datos.length.toLocaleString() + " ops";
    document.getElementById("leg-min").textContent  = fmt(mn);
    document.getElementById("leg-max").textContent  = fmt(mx);
    extremosActuales = {{ min:minCaptacion, max:maxCaptacion }};
    actualizarTarjetaExtremo("min", minCaptacion);
    actualizarTarjetaExtremo("max", maxCaptacion);
}}

function limpiarCapas() {{
    if (layerHeat)     {{ map.removeLayer(layerHeat);     layerHeat = null; }}
    if (layerPuntos)   {{ map.removeLayer(layerPuntos);   layerPuntos = null; }}
    if (layerClusters) {{ map.removeLayer(layerClusters); layerClusters = null; }}
    if (layerVoronoi)  {{ map.removeLayer(layerVoronoi);  layerVoronoi = null; }}
    if (layerExtremos) {{ map.removeLayer(layerExtremos); layerExtremos = null; }}
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

        layerHeat = L.heatLayer(puntos, {{
            radius: 25, blur: 20, maxZoom: 15, max: 1.0,
            gradient: {{ 0.0:"#00FF00", 0.4:"#FFFF00", 0.7:"#FFA500", 1.0:"#FF0000" }}
        }}).addTo(map);

    }} else {{
        // --- DENSIDAD: normalizar por percentil 95 para evitar saturación ---
        puntos = datos.map(d => [d.lat, d.lng, 1]);
        document.getElementById("leyenda-titulo").textContent = "Actividad / Operaciones";
        document.getElementById("leyenda-grad").style.background =
            "linear-gradient(to right,#edf8fb,#b2e2e2,#66c2a4,#2ca25f,#006d2c)";
        document.getElementById("leg-mid").textContent = "densidad media";
        document.getElementById("leg-min").textContent = "Poca actividad";
        document.getElementById("leg-max").textContent = "Mucha actividad";

        // Calcular un radio dinámico: menos radio = más granularidad, se ven los puntos
        const zoom = map.getZoom();
        const radioAdaptado = Math.max(8, Math.min(18, zoom * 1.2));
        
        layerHeat = L.heatLayer(puntos, {{
            radius:  radioAdaptado,
            blur:    radioAdaptado * 0.5,
            maxZoom: 15,
            max:     1.8,   // ← clave: satura recién con 3 puntos superpuestos, no 1
            minOpacity: 0.08,
        gradient: {{
            0.0: "rgba(0,0,255,0)",
            0.2: "rgba(0,255,255,0.3)",
            0.4: "rgba(0,255,0,0.4)",
            0.6: "rgba(255,255,0,0.5)",
            0.8: "rgba(255,165,0,0.6)",
            1.0: "rgba(255,0,0,0.5)"
        }}
        }}).addTo(map);
    }}
}}

function esc(valor) {{
    return String(valor ?? "").replace(/[&<>'"]/g, caracter => ({{
        "&":"&amp;", "<":"&lt;", ">":"&gt;", "'":"&#39;", '"':"&quot;"
    }})[caracter]);
}}

function popupCaptacion(d, extremo) {{
    const etiqueta = extremo === "min" ? "Menor precio filtrado" : "Mayor precio filtrado";
    const cierre = d.transaccion === "Alquiler" ? "Cierre mensual" : "Cierre de venta";
    return `
        <div class="price-popup">
            <span class="badge">${{etiqueta}}</span>
            <div class="tt-titulo" style="margin-top:7px">${{esc(d.tipo)}} · ${{esc(d.transaccion)}}</div>
            <div class="tt-fila"><span class="tt-lbl">Captación</span><span class="tt-val">${{esc(d.id)}}</span></div>
            <div class="tt-fila"><span class="tt-lbl">MLSID</span><span class="tt-val">${{esc(d.mlsid)}}</span></div>
            <div class="tt-fila"><span class="tt-lbl">${{cierre}}</span><span class="tt-val">${{fmt(d.precio)}}</span></div>
            <div class="tt-fila"><span class="tt-lbl">Publicación</span><span class="tt-val">${{fmt(d.precio_publicacion)}}</span></div>
            <div class="tt-fila"><span class="tt-lbl">Precio/m²</span><span class="tt-val">${{fmtM2(d.precio_m2)}}</span></div>
            <div class="tt-fila"><span class="tt-lbl">Ciudad</span><span class="tt-val">${{esc(d.ciudad)}}</span></div>
            <div class="tt-fila"><span class="tt-lbl">Zona</span><span class="tt-val">${{d.cluster >= 0 ? "Cluster " + d.cluster : "—"}}</span></div>
        </div>`;
}}

function crearMarcadorExtremo(captacion, tipo) {{
    const esMin = tipo === "min";
    const icono = L.divIcon({{
        className:"",
        html:`<div class="price-flag ${{esMin ? "low" : "high"}}"><span>${{esMin ? "MENOR" : "MAYOR"}}</span><strong>${{fmt(captacion.precio)}}</strong></div>`,
        iconSize:null,
        iconAnchor:[42,16],
    }});
    return L.marker([captacion.lat, captacion.lng], {{icon:icono, zIndexOffset:1000}})
        .bindPopup(popupCaptacion(captacion, tipo), {{maxWidth:300}});
}}

function renderExtremos() {{
    layerExtremos = L.layerGroup();
    marcadoresExtremos = {{ min:null, max:null }};
    if (!extremosActuales.min || !extremosActuales.max) return;

    const minMarker = crearMarcadorExtremo(extremosActuales.min, "min");
    minMarker.addTo(layerExtremos);
    marcadoresExtremos.min = minMarker;

    const mismaCaptacion = extremosActuales.min.id === extremosActuales.max.id;
    if (mismaCaptacion) {{
        marcadoresExtremos.max = minMarker;
    }} else {{
        const maxMarker = crearMarcadorExtremo(extremosActuales.max, "max");
        maxMarker.addTo(layerExtremos);
        marcadoresExtremos.max = maxMarker;
    }}
    layerExtremos.addTo(map);
}}

function enfocarExtremo(tipo) {{
    const captacion = extremosActuales[tipo];
    const marker = marcadoresExtremos[tipo];
    if (!captacion || !marker) return;
    const reduceMovimiento = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMovimiento) map.setView([captacion.lat, captacion.lng], 16);
    else map.flyTo([captacion.lat, captacion.lng], 16, {{duration:.8}});
    window.setTimeout(() => marker.openPopup(), reduceMovimiento ? 0 : 650);
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
                <div class="tt-fila"><span class="tt-lbl">Captación</span><span class="tt-val">${{d.id}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">MLSID</span><span class="tt-val">${{d.mlsid}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Ciudad</span><span class="tt-val">${{d.ciudad}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Cierre</span><span class="tt-val">${{fmt(d.precio)}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Publicación</span><span class="tt-val">${{fmt(d.precio_publicacion)}}</span></div>
                <div class="tt-fila"><span class="tt-lbl">Precio/m²</span><span class="tt-val">${{fmtM2(d.precio_m2)}}</span></div>
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
                <div class="tt-fila"><span class="tt-lbl">Precio/m² prom.</span><span class="tt-val">${{fmtM2(m2p)}}</span></div>
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
    renderExtremos();
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
    ruta = f"data/{nombre}"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)

    log.info("\nMapa guardado en %s", ruta)
    webbrowser.open(f"file:///{os.path.abspath(ruta)}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("INTRAMAX — Mapa de Calor de Precios v2")
    log.info("=" * 55)

    # Sin argumento → Bolivia completo
    # Con argumento → ciudad específica
    # Ejemplos:
    #   python -m etl.clustering.visualizar_heatmap
    #   python -m etl.clustering.visualizar_heatmap "Santa Cruz de la Sierra"
    #   python -m etl.clustering.visualizar_heatmap Cochabamba

    if len(sys.argv) > 1:
        ciudad = " ".join(sys.argv[1:])
        log.info(f"  Ciudad: {ciudad}")
        generar_mapa(ciudad)
    else:
        log.info("  Modo: Bolivia completo")
        generar_mapa(None)
