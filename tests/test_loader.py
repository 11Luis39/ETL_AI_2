from datetime import date

import numpy as np
import pandas as pd

from etl.loader import _limpiar_valor, _preparar_fila


def test_preparar_fila_no_sobrescribe_m2_terreno_calculado():
    fila = _preparar_fila(
        {
            "id_propiedad": "T-1",
            "tipo_transaccion": "Venta",
            "m2_terreno": 1_500,
            "land_m2": 0,
            "precio_cierre": 45_000,
            "fecha_venta": pd.Timestamp("2026-01-10"),
        }
    )

    assert fila["m2_terreno"] == 1_500
    assert fila["precio_venta"] == 45_000
    assert fila["precio_alquiler_mes"] is None
    assert fila["fecha_venta"] == date(2026, 1, 10)


def test_preparar_fila_separa_precio_de_alquiler():
    fila = _preparar_fila(
        {
            "id_propiedad": "A-1",
            "tipo_transaccion": "Alquiler",
            "precio_cierre": 2_500,
        }
    )

    assert fila["precio_venta"] is None
    assert fila["precio_alquiler_mes"] == 2_500


def test_limpiar_valor_convierte_tipos_numpy_y_no_finitos():
    assert _limpiar_valor(np.int64(7)) == 7
    assert isinstance(_limpiar_valor(np.int64(7)), int)
    assert _limpiar_valor(np.inf) is None
