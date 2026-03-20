import logging
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Ciudades y bbox
# ------------------------------------------------------------
CIUDADES_ETL = {
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

def get_crm_engine():
    url = (
        f"mysql+pymysql://{os.getenv('CRM_USERNAME')}:{os.getenv('CRM_PASSWORD')}"
        f"@{os.getenv('CRM_HOST')}:{os.getenv('CRM_PORT')}/{os.getenv('CRM_DATABASE')}"
    )
    return create_engine(url)

# ------------------------------------------------------------
# Query parametrizada — funciona para venta y alquiler
# ------------------------------------------------------------
QUERY_CRM = text("""
    SELECT
        l.id                            AS id_propiedad,
        l.MLSID                         AS mlsid,
        l.date_of_listing,
        l.cancellation_date,
        l.contract_end_date,

        a.name                          AS segmento,
        ltt.name                        AS transaction_type,
        sl.name                         AS status,
        sp.name                         AS subtipo_original,
        pc.name_properties_categories   AS categoria_propiedad,
        stp.name_state_properties       AS estado_propiedad,
        ci.name                         AS ciudad,

        li.construction_area_m,
        li.total_area,
        li.land_m2,
        li.number_bedrooms              AS dormitorios,
        li.number_bathrooms             AS banos,
        li.parking_slots                AS estacionamientos,
        li.year_construction,
        lp.amount                       AS precio_publicacion,
        loc.latitude,
        loc.longitude,

        t.sold_date,
        t.current_listing_price         AS precio_cierre

    FROM listings l
    LEFT JOIN listing_transaction_types ltt ON l.transaction_type_id = ltt.id
    LEFT JOIN status_listings sl            ON l.status_listing_id = sl.id
    LEFT JOIN listings_information li       ON li.listing_id = l.id
    LEFT JOIN subtype_properties sp         ON li.subtype_property_id = sp.id
    LEFT JOIN state_properties stp          ON li.state_property_id = stp.id
    LEFT JOIN properties_category pc        ON li.property_category_id = pc.id
    LEFT JOIN listing_prices lp             ON lp.listing_id = l.id
    LEFT JOIN locations loc                 ON loc.listing_id = l.id
    LEFT JOIN cities ci                     ON loc.city_id = ci.id
    LEFT JOIN areas a                       ON a.id = l.area_id
    LEFT JOIN transactions t ON t.listing_id = l.id
        AND t.transaction_type_id = :ltt_id
        AND t.transaction_status_id IN (2, 5)

    WHERE ltt.id = :ltt_id
      AND l.status_listing_id = :status_id
      AND ci.name = :ciudad
      AND loc.latitude  BETWEEN :lat_min AND :lat_max
      AND loc.longitude BETWEEN :lng_min AND :lng_max
      AND YEAR(t.sold_date) IN (2025, 2026)
    ORDER BY l.id DESC
""")

# Tipos a extraer: Venta y Alquiler
TIPOS_TRANSACCION = [
    {"ltt_id": 1, "status_id": 8, "label": "Venta"},
    {"ltt_id": 2, "status_id": 7, "label": "Alquiler"},
]

# ------------------------------------------------------------
# Extracción principal
# ------------------------------------------------------------
def extraer_datos_crm() -> pd.DataFrame:
    engine = get_crm_engine()
    dfs    = []

    for tipo in TIPOS_TRANSACCION:
        log.info(f"\n  Extrayendo {tipo['label']}...")

        for ciudad, bbox in CIUDADES_ETL.items():
            try:
                with engine.connect() as conn:
                    df_ciudad = pd.read_sql(QUERY_CRM, conn, params={
                        "ltt_id":    tipo["ltt_id"],
                        "status_id": tipo["status_id"],
                        "ciudad":    ciudad,
                        "lat_min":   bbox["lat"][0],
                        "lat_max":   bbox["lat"][1],
                        "lng_min":   bbox["lng"][0],
                        "lng_max":   bbox["lng"][1],
                    })

                if len(df_ciudad) > 0:
                    df_ciudad["tipo_transaccion"] = tipo["label"]
                    log.info(f"    → {ciudad}: {len(df_ciudad)} {tipo['label'].lower()}s")
                    dfs.append(df_ciudad)

            except Exception as e:
                log.error(f"    Error extrayendo {tipo['label']} en {ciudad}: {e}")
                continue

    if not dfs:
        log.error("No se extrajeron datos de ninguna ciudad")
        return pd.DataFrame()

    df_total   = pd.concat(dfs, ignore_index=True)
    ventas     = (df_total["tipo_transaccion"] == "Venta").sum()
    alquileres = (df_total["tipo_transaccion"] == "Alquiler").sum()

    log.info(f"\nExtracción completa: {len(df_total)} filas")
    log.info(f"  Ventas:     {ventas}")
    log.info(f"  Alquileres: {alquileres}")

    return df_total