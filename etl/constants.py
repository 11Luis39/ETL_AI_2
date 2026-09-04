"""Valores de negocio compartidos por extracción y visualización."""

CIUDADES_ETL = {
    "Santa Cruz de la Sierra": {"lat": (-18.5, -17.0), "lng": (-64.0, -62.5)},
    "Cochabamba": {"lat": (-17.8, -17.2), "lng": (-66.5, -65.8)},
    "La Paz": {"lat": (-16.8, -16.3), "lng": (-68.3, -67.8)},
    "Porongo": {"lat": (-17.9, -17.5), "lng": (-63.6, -63.2)},
    "El Alto": {"lat": (-16.6, -16.4), "lng": (-68.3, -68.0)},
    "Oruro": {"lat": (-18.1, -17.8), "lng": (-67.2, -67.0)},
    "Tiquipaya": {"lat": (-17.4, -17.2), "lng": (-66.3, -66.1)},
    "Sacaba": {"lat": (-17.5, -17.3), "lng": (-65.9, -65.7)},
    "Sucre": {"lat": (-19.2, -18.9), "lng": (-65.4, -65.1)},
    "La Guardia": {"lat": (-17.9, -17.7), "lng": (-63.4, -63.2)},
    "Warnes": {"lat": (-17.6, -17.4), "lng": (-63.3, -63.1)},
    "Quillacollo": {"lat": (-17.5, -17.3), "lng": (-66.3, -66.1)},
    "Samaipata": {"lat": (-18.3, -18.1), "lng": (-63.9, -63.7)},
    "Cotoca": {"lat": (-17.9, -17.7), "lng": (-63.1, -62.9)},
    "Potosí": {"lat": (-19.7, -19.4), "lng": (-65.9, -65.6)},
}

TIPOS_TRANSACCION = (
    {"ltt_id": 1, "status_id": 8, "label": "Venta"},
    {"ltt_id": 2, "status_id": 7, "label": "Alquiler"},
)
