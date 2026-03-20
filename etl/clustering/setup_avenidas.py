import os
import logging
import sys
import osmnx as ox
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Clasificación completa de vías por categoría y zona
# ------------------------------------------------------------

VIAS = {

    # 1 — Vías de Conexión Nacional y Regional
    "conexion_nacional": {
        "label":  "Conexión Nacional/Regional",
        "color":  "#DC2626",   # rojo intenso
        "weight": 5,
        "vias": [
            {"nombre": "Avenida Grigotá", "exacto": True},
            {"nombre": "Doble Vía Santa Cruz - La Guardia", "exacto": True},
            {"nombre": "Avenida Virgen de Cotoca", "exacto": True},
            {"nombre": "Avenida Cristo Redentor", "exacto": True},
            {"nombre": "Avenida G77", "exacto": True},
        ],
    },

    # 2 — Anillos principales (1ro al 4to)
    "anillos_principales": {
        "label":  "Anillos Principales (1ro–4to)",
        "color":  "#EA580C",
        "weight": 4,
        "vias": [
            {"nombre": "Avenida Uruguay",               "exacto": True},
            {"nombre": "Avenida Cañoto",                "exacto": True},
            {"nombre": "Avenida Irala",                 "exacto": True},
            {"nombre": "Avenida Argomosa",              "exacto": True},
            {"nombre": "Avenida El Trompillo",          "exacto": True},
            {"nombre": "Avenida Viedma",                "exacto": True},
            {"nombre": "Avenida Santa Cruz",            "exacto": True},
            {"nombre": "Avenida Cristóbal de Mendoza",  "exacto": True},
            {"nombre": "Avenida 26 de febrero",         "exacto": True},
            "Tercer Anillo",                                      
            {"nombre": "Avenida Roque Aguilera",        "exacto": True},
            {"nombre": "Avenida Noel Kempff",           "exacto": True},
            {"nombre": "Avenida Noel Kempff Mercado",   "exacto": True,
             "bbox": {
                "lat_min": -17.81, "lat_max": -17.74, 
                "lng_min": -63.21, "lng_max": -63.15,
            }},
            {"nombre": "Avenida Juan Pablo II", "exacto": True, "bbox": {
                "lat_min": -17.81, "lat_max": -17.79,
                "lng_min": -63.17, "lng_max": -63.15,
            }},
            "Cuarto Anillo",
            {"nombre": "Avenida Antonio Vaca Diez",     "exacto": True},
            {"nombre": "Avenida Marcelo Terceros Bánzer", "exacto": True},
        ],
    },

    # 3 — Anillos de expansión (5to al 8vo)
    "anillos_expansion": {
        "label":  "Anillos Expansión (5to–8vo)",
        "color":  "#F59E0B",   # amarillo
        "weight": 3,
        "vias": [
            "Quinto Anillo",
            "Sexto Anillo",
            "Séptimo Anillo",
            "Octavo Anillo",
            "Noveno Anillo",
        ],
    },

    # 4 — Avenidas Primarias (estructurantes)


"primarias": {
        "label":  "Avenidas Primarias",
        "color":  "#2563EB",   # azul
        "weight": 4,
        "vias": [
            {"nombre": "Avenida San Martín", "exacto": True},
            {"nombre": "Avenida Busch", "exacto": True},
            {"nombre": "Avenida Beni", "exacto": True},
            {"nombre": "Avenida Alemania", "exacto": True},
            {"nombre": "Avenida Mutualista", "exacto": True},
            {"nombre": "Avenida Santos Dumont", "exacto": True},
            {"nombre": "Avenida Roca y Coronado", "exacto": True},
            {"nombre": "Avenida Piraí", "exacto": True},
            {"nombre": "Avenida San Aurelio", "exacto": True},
            {"nombre": "Avenida Irala", "exacto": True},
            {"nombre": "Avenida Cañoto", "exacto": True},
            {"nombre": "Avenida Paraguá", "exacto": True},
            {"nombre": "Avenida Guapay", "exacto": True},
            {"nombre": "Avenida Brasil", "exacto": True},
            {"nombre": "Avenida Intermodal", "exacto": True},
            {"nombre": "Avenida 3 Pasos al frente", "exacto": True},
            {"nombre": "Avenida Centenario", "exacto": True},
            {"nombre": "Avenida La Salle", "exacto": True},
            {"nombre": "Radial 21", "exacto": True},
            {"nombre": "Avenida El Palmar", "exacto": True},
            {"nombre": "Avenida Ovidio Barbery Justiniano", "exacto": True},
            {"nombre": "Radial 17 1/2", "exacto": True},
            {"nombre": "Radial 27", "exacto": True},
            {"nombre": "Avenida Hernando Sanabria", "exacto": True},
            {"nombre": "Avenida Landivar", "exacto": True},
            {"nombre": "Avenida Prefecto Rivas", "exacto": True},
            {"nombre": "Avenida Escuadrón Velasco", "exacto": True},
            {"nombre": "Avenida Monseñor Rivero", "exacto": True},
            {"nombre": "Avenida Trinidad", "exacto": True},
            {"nombre": "Avenida Suárez Arana", "exacto": True},
            {"nombre": "Avenida Charcas", "exacto": True},
            {"nombre": "Avenida Melchor Pinto", "exacto": True},
            {"nombre": "Avenida Argentina", "exacto": True},
            {"nombre": "Avenida Francisco Velarde", "exacto": True},
            {"nombre": "Avenida Omar Chávez Ortiz", "exacto": True},
            {"nombre": "Radial 13", "exacto": True},
            {"nombre": "Avenida Coronel Maximiliano España", "exacto": True},
            {"nombre": "Avenida El Trillo", "exacto": True},
            {"nombre": "Avenida Moscú", "exacto": True},
            {"nombre": "Avenida Radial 10", "exacto": True},
            {"nombre": "Avenida Internacional", "exacto": True},
            {"nombre": "Radial 16", "exacto": True},
            {"nombre": "Radial 26", "exacto": True},
            {"nombre": "Avenida José Benjamín Burela Justiniano", "exacto": True},
            {"nombre": "Avenida Capitán Alfredo Higazy", "exacto": True},
            {"nombre": "Avenida Marcelo Quiroga Santa Cruz", "exacto": True},
            {"nombre": "Avenida Adolfo Román Hijo", "exacto": True},
            {"nombre": "Avenida Mariscal Santa Cruz", "exacto": True},
            {"nombre": "Avenida Totaí", "exacto": True},
            {"nombre": "Avenida Miguel de Cervantes", "exacto": True},
            {"nombre": "Avenida Pedro Casais", "exacto": True},
            {"nombre": "Avenida Olímpica", "exacto": True},
            {"nombre": "Avenida hilandería", "exacto": True},
            {"nombre": "Calle Diamante", "exacto": True},
        ],
    },

    # 5 — Plan 3000
    "plan_3000": {
        "label":  "Plan 3.000 (D-8)",
        "color":  "#7C3AED",   # violeta
        "weight": 3,
        "vias": [
            "Paurito",
            "Mechero",
            "La Campana",
            "Plan Tres Mil",
            "Monseñor Nicolás Castellanos Franco",
            "Palmar Viruez",
            "El Quior"
        ],
    },

    # 6 — Villa Primero de Mayo
    "villa": {
        "label":  "Villa 1ro de Mayo (D-7)",
        "color":  "#059669",   # verde
        "weight": 3,
        "vias": [
            "Cumavi",
            "Tres Pasos al Frente",
            "16 de Julio",
            "General Campero",
            "Arroyito",
        ],
    },

    # 7 — Pampa de la Isla
    "pampa": {
        "label":  "Pampa de la Isla (D-6)",
        "color":  "#0891B2",   # celeste
        "weight": 3,
        "vias": [
            "Virgen de Luján",
            "Montecristo",
            "Las Orquídeas",
        ],
    },

    # 8 — Los Lotes / Zona Sur
    "zona_sur": {
        "label":  "Los Lotes / Zona Sur (D-9/D-12)",
        "color":  "#BE185D",   # rosa oscuro
        "weight": 3,
        "vias": [
            {"nombre": "Avenida Prolongacion Bolivia", "exacto": True},
            {"nombre": "Avenida El Palmar del Oratorio", "exacto": True},
            {"nombre": "Avenida Blooming", "exacto": True},
            {"nombre": "Avenida El Fuerte", "exacto": True},
            {"nombre": "Avenida 4 de Octubre", "exacto": True},
        ],
    },
}


# ------------------------------------------------------------
# Descargar red vial completa de Santa Cruz
# ------------------------------------------------------------
def descargar_red_vial():
    ruta_cache = "data/geo/red_vial_scz.gpkg"

    if os.path.exists(ruta_cache):
        log.info(f"Red vial ya descargada → cargando desde cache: {ruta_cache}")
        edges = gpd.read_file(ruta_cache)
        return edges

    log.info("Descargando red vial de Santa Cruz desde OpenStreetMap...")
    log.info("Esto puede tardar 3-5 minutos la primera vez...")

    G = ox.graph_from_place(
        "Santa Cruz de la Sierra, Bolivia",
        network_type="all",        # incluye peatonal y vehicular
        retain_all=False,
    )

    _, edges = ox.graph_to_gdfs(G)
    log.info(f"  → {len(edges):,} segmentos viales descargados")

    # Normalizar columna name
    edges["name_norm"] = edges["name"].apply(
        lambda x: x if isinstance(x, str)
        else (", ".join(x) if isinstance(x, list) else "")
    )

    os.makedirs("data/geo", exist_ok=True)
    edges.to_file(ruta_cache, driver="GPKG")
    log.info(f"  → Red vial guardada en {ruta_cache}")

    return edges


# ------------------------------------------------------------
# Filtrar y clasificar vías importantes
# ------------------------------------------------------------
def filtrar_vias(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log.info("\nFiltrando vías clasificadas...")

    def normalizar_nombres(val):
        if isinstance(val, list):
            return [str(v).strip().lower() for v in val]
        elif isinstance(val, str):
            partes = val.replace(";", ",").split(",")
            return [p.strip().lower() for p in partes]
        return []

    edges["nombres_lista"] = edges["name"].apply(normalizar_nombres)

    # Pre-calcular centroides en WGS84 una sola vez
    if edges.crs and edges.crs.to_epsg() != 4326:
        edges_wgs = edges.to_crs("EPSG:4326")
    else:
        edges_wgs = edges
    centroides = edges_wgs.geometry.centroid

    resultados        = []
    total_encontradas = 0
    total_no_encontradas = 0

    for categoria, info in VIAS.items():
        for via in info["vias"]:

            if isinstance(via, dict):
                nombre = via["nombre"]
                exacto = via.get("exacto", False)
                bbox   = via.get("bbox", None)      # ← None si no tiene bbox
            else:
                nombre = via
                exacto = False
                bbox   = None

            nombre_lower = nombre.strip().lower()

            # 1 — Filtro por nombre
            if exacto:
                mask = edges["nombres_lista"].apply(
                    lambda lista: nombre_lower in lista
                )
            else:
                mask = edges["nombres_lista"].apply(
                    lambda lista: any(nombre_lower in n for n in lista)
                )

            # 2 — Filtro por bbox SOLO si está definido
            if bbox is not None:
                mask_bbox = (
                    (centroides.y >= bbox["lat_min"]) &
                    (centroides.y <= bbox["lat_max"]) &
                    (centroides.x >= bbox["lng_min"]) &
                    (centroides.x <= bbox["lng_max"])
                )
                mask = mask & mask_bbox  # ← solo se aplica si hay bbox

            segs = edges[mask].copy()

            if len(segs) > 0:
                segs["categoria"]  = categoria
                segs["label"]      = info["label"]
                segs["color"]      = info["color"]
                segs["weight"]     = info["weight"]
                segs["nombre_via"] = nombre
                resultados.append(segs)
                modo = "exacto" if exacto else "parcial"
                bbox_info = " +bbox" if bbox else ""
                log.info(f"  ✓ [{modo}{bbox_info}] {nombre:40} → {len(segs):4} segs")
                total_encontradas += 1
            else:
                log.warning(f"  ✗ No encontrada: {nombre}")
                total_no_encontradas += 1

    log.info(f"\n  Encontradas:    {total_encontradas}")
    log.info(f"  No encontradas: {total_no_encontradas}")

    if not resultados:
        log.error("No se encontró ninguna vía.")
        return None

    df = pd.concat(resultados, ignore_index=True)
    df = df.drop_duplicates(subset=["geometry"])

    ruta = "data/geo/vias_clasificadas.gpkg"
    df.to_file(ruta, driver="GPKG")
    log.info(f"\n  → {len(df):,} segmentos guardados en {ruta}")

    return df

# ------------------------------------------------------------
# Resumen por categoría
# ------------------------------------------------------------
def imprimir_resumen(df: gpd.GeoDataFrame):
    log.info("\n" + "="*60)
    log.info("RESUMEN POR CATEGORÍA")
    log.info("="*60)

    resumen = df.groupby("label").size().reset_index(name="segmentos")
    for _, row in resumen.iterrows():
        log.info(f"  {row['label']:45} {row['segmentos']:5} segmentos")

    log.info(f"\n  TOTAL: {len(df):,} segmentos clasificados")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    log.info("="*60)
    log.info("INTRAMAX — Setup Vías Santa Cruz de la Sierra")
    log.info("="*60)

    os.makedirs("data/geo", exist_ok=True)

    edges = descargar_red_vial()
    df_vias = filtrar_vias(edges)

    if df_vias is not None:
        imprimir_resumen(df_vias)
        log.info("\n✓ Setup completado exitosamente")
        log.info("  Archivos generados:")
        log.info("  - data/geo/red_vial_scz.gpkg       (red vial completa)")
        log.info("  - data/geo/vias_clasificadas.gpkg  (vías clasificadas)")
        log.info("\nAhora podés correr:")
        log.info("  python etl/clustering/visualizar_avenidas.py")
    else:
        log.error("Setup falló — verificar conexión a internet")