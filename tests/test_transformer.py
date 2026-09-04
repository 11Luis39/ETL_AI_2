import numpy as np
import pandas as pd

from etl.transformer import _asignar_clusters_multiciudad, calcular_precio_m2


def test_precio_m2_usa_publicacion_si_el_cierre_es_nulo():
    df = pd.DataFrame(
        {
            "subtipo_original": ["Departamento", "Casa"],
            "construction_area_m": [50, 100],
            "total_area": [80, 200],
            "precio_cierre": [np.nan, 180_000],
            "precio_publicacion": [50_000, 200_000],
        }
    )

    resultado = calcular_precio_m2(df)

    assert resultado.tolist() == [1_000.0, 900.0]


def test_precio_m2_no_divide_por_superficie_invalida():
    df = pd.DataFrame(
        {
            "subtipo_original": ["Departamento"],
            "construction_area_m": [0],
            "total_area": [100],
            "precio_cierre": [10_000],
            "precio_publicacion": [12_000],
        }
    )

    assert calcular_precio_m2(df).isna().all()


def test_asignar_clusters_respeta_centroides_de_cada_ciudad():
    propiedades = pd.DataFrame(
        {
            "ciudad": ["Ciudad A", "Ciudad A", "Ciudad B"],
            "latitude": [0.1, 9.8, 50.2],
            "longitude": [0.1, 9.8, 50.2],
        }
    )
    centroides = pd.DataFrame(
        {
            "cluster_id": [0, 1, 0],
            "ciudad": ["Ciudad A", "Ciudad A", "Ciudad B"],
            "centroide_lat": [0.0, 10.0, 50.0],
            "centroide_lng": [0.0, 10.0, 50.0],
        }
    )

    resultado = _asignar_clusters_multiciudad(
        propiedades,
        centroides=centroides,
    )

    assert resultado["cluster_zona"].tolist() == [0, 1, 0]
