"""Extracción de operaciones cerradas desde el CRM."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import Engine, bindparam, text

from .config import get_crm_engine
from .constants import CIUDADES_ETL, TIPOS_TRANSACCION

log = logging.getLogger(__name__)


# La comparación directa sobre sold_date permite que MySQL use su índice.
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
    LEFT JOIN listing_prices lp             ON lp.listing_id = l.id
    LEFT JOIN locations loc                 ON loc.listing_id = l.id
    LEFT JOIN cities ci                     ON loc.city_id = ci.id
    LEFT JOIN areas a                       ON a.id = l.area_id
    JOIN transactions t ON t.listing_id = l.id
        AND t.transaction_type_id = :ltt_id
        AND t.transaction_status_id IN (2, 5)

    WHERE ltt.id = :ltt_id
      AND l.status_listing_id = :status_id
      AND ci.name IN :ciudades
      AND t.sold_date >= :fecha_desde
""").bindparams(bindparam("ciudades", expanding=True))


def _filtrar_bounding_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo coordenadas que caen dentro del rango de su ciudad."""

    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")
    valida = pd.Series(False, index=df.index)
    for ciudad, bbox in CIUDADES_ETL.items():
        valida |= (
            df["ciudad"].eq(ciudad)
            & latitude.between(*bbox["lat"])
            & longitude.between(*bbox["lng"])
        )
    return df.loc[valida].copy()


def extraer_datos_crm(engine: Engine | None = None) -> pd.DataFrame:
    """Extrae ventas y alquileres de los últimos tres años calendario.

    Se acepta un ``engine`` opcional para facilitar las pruebas. Durante una
    ejecución real hace solo una consulta por tipo de transacción y valida en
    bloque los rangos geográficos de cada ciudad.
    """

    engine = engine or get_crm_engine()
    fecha_desde = date(date.today().year - 2, 1, 1)
    dataframes: list[pd.DataFrame] = []

    with engine.connect() as connection:
        for transaccion in TIPOS_TRANSACCION:
            log.info("\n  Extrayendo %s...", transaccion["label"])
            params = {
                "ltt_id": transaccion["ltt_id"],
                "status_id": transaccion["status_id"],
                "ciudades": list(CIUDADES_ETL),
                "fecha_desde": fecha_desde,
            }

            try:
                df_transaccion = pd.read_sql(QUERY_CRM, connection, params=params)
            except Exception:
                log.exception("    Error extrayendo %s", transaccion["label"])
                continue

            df_transaccion = _filtrar_bounding_boxes(df_transaccion)
            if df_transaccion.empty:
                continue

            df_transaccion["tipo_transaccion"] = transaccion["label"]
            for ciudad, cantidad in df_transaccion["ciudad"].value_counts().items():
                log.info("    → %s: %s registros", ciudad, cantidad)
            dataframes.append(df_transaccion)

    if not dataframes:
        log.error("No se extrajeron datos de ninguna ciudad")
        return pd.DataFrame()

    df_total = pd.concat(dataframes, ignore_index=True)
    ventas = (df_total["tipo_transaccion"] == "Venta").sum()
    alquileres = (df_total["tipo_transaccion"] == "Alquiler").sum()

    log.info("\nExtracción completa: %s filas", len(df_total))
    log.info("  Ventas:     %s", ventas)
    log.info("  Alquileres: %s", alquileres)
    return df_total
