import pandas as pd

from etl.extractor import _filtrar_bounding_boxes


def test_filtrar_bounding_boxes_respeta_la_ciudad_de_cada_coordenada():
    df = pd.DataFrame(
        {
            "ciudad": [
                "Santa Cruz de la Sierra",
                "Santa Cruz de la Sierra",
                "Cochabamba",
                "Ciudad no configurada",
            ],
            "latitude": [-17.78, -30.0, -17.39, -17.78],
            "longitude": [-63.18, -63.18, -66.16, -63.18],
        }
    )

    resultado = _filtrar_bounding_boxes(df)

    assert resultado.index.tolist() == [0, 2]
