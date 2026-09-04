import pandas as pd

from etl.cleaner import _remover_outliers, limpiar_datos


def propiedad(**cambios):
    base = {
        "id_propiedad": "P-1",
        "mlsid": "MLS-1",
        "subtipo_original": "Casa",
        "tipo_transaccion": "Venta",
        "segmento": "Residencial",
        "latitude": -17.78,
        "longitude": -63.18,
        "construction_area_m": 150,
        "total_area": 300,
        "land_m2": 300,
        "dormitorios": 3,
        "banos": 2,
        "precio_publicacion": 100_000,
        "precio_cierre": 95_000,
    }
    base.update(cambios)
    return base


def test_limpieza_conserva_ultima_fila_duplicada_y_reporta_la_anterior():
    df = pd.DataFrame(
        [
            propiedad(precio_publicacion=90_000),
            propiedad(precio_publicacion=100_000),
        ]
    )

    limpios, excluidos = limpiar_datos(df)

    assert len(limpios) == 1
    assert limpios.iloc[0]["precio_publicacion"] == 100_000
    assert limpios.iloc[0]["m2_construidos"] == 150
    assert limpios.iloc[0]["m2_terreno"] == 300
    assert "duplicado" in excluidos["motivo"].tolist()


def test_limpieza_reporta_todos_los_campos_requeridos_invalidos():
    df = pd.DataFrame(
        [
            propiedad(
                id_propiedad="P-2",
                subtipo_original="Departamento",
                dormitorios=0,
                banos=None,
            )
        ]
    )

    limpios, excluidos = limpiar_datos(df)

    assert limpios.empty
    assert set(excluidos["motivo"]) == {
        "dormitorios nulo o menor/igual a 0",
        "baños nulo o menor/igual a 0",
    }


def test_limpieza_descarta_coordenadas_invalidas_y_precios_bajos():
    df = pd.DataFrame(
        [
            propiedad(id_propiedad="SIN-GEO", latitude=999),
            propiedad(
                id_propiedad="BARATA",
                tipo_transaccion="Alquiler",
                precio_publicacion=50,
            ),
        ]
    )

    limpios, excluidos = limpiar_datos(df)

    assert limpios.empty
    assert set(excluidos["motivo"]) == {
        "coordenadas ausentes o inválidas",
        "precio menor al mínimo de alquiler",
    }


def test_outliers_se_calculan_por_tipo_y_transaccion_sin_iterar_filas():
    filas = [propiedad(id_propiedad=f"P-{i}", precio_cierre=100.0) for i in range(12)]
    filas.append(propiedad(id_propiedad="EXTREMO", precio_cierre=100_000.0))
    df = pd.DataFrame(filas)
    df["tipo_propiedad"] = "Casa"

    limpio, excluidos = _remover_outliers(df, "precio_publicacion")

    assert "EXTREMO" not in limpio["id_propiedad"].tolist()
    assert excluidos[0]["motivo"] == "outlier precio_cierre"
