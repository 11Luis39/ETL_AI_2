from pathlib import Path

import pandas as pd

from etl.clustering import visualizar_heatmap as heatmap


def operaciones() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_propiedad": ["CAP-VENTA", "CAP-ALQUILER"],
            "mlsid": ["MLS-100", "MLS-200"],
            "tipo_propiedad": ["Casa", "Departamento"],
            "tipo_transaccion": ["Venta", "Alquiler"],
            "cluster_zona": [0, 0],
            "ciudad": ["Santa Cruz de la Sierra", "Santa Cruz de la Sierra"],
            "latitude": [-17.78, -17.79],
            "longitude": [-63.18, -63.19],
            "precio_publicacion": [120_000, 700],
            "precio_venta": [110_000, None],
            "precio_alquiler_mes": [None, 650],
            "precio_m2": [730, 8],
            "m2_construidos": [150, 80],
            "m2_terreno": [300, 0],
            "dormitorios": [3, 2],
            "banos": [2, 1],
            "anio_venta": [2026, 2026],
        }
    )


def test_preparar_registros_identifica_captacion_y_precios_usd():
    registros = heatmap.preparar_registros(operaciones())

    assert registros[0]["id"] == "CAP-VENTA"
    assert registros[0]["precio"] == 110_000
    assert registros[0]["precio_publicacion"] == 120_000
    assert registros[1]["id"] == "CAP-ALQUILER"
    assert registros[1]["precio"] == 650


def test_mapa_incluye_controles_para_captaciones_extremas(
    tmp_path: Path,
    monkeypatch,
):
    centroides = pd.DataFrame(
        {
            "cluster_id": [0],
            "ciudad": ["Santa Cruz de la Sierra"],
            "centroide_lat": [-17.785],
            "centroide_lng": [-63.185],
            "total_propiedades": [2],
        }
    )
    monkeypatch.setattr(heatmap, "cargar_datos", lambda ciudad=None: operaciones())
    monkeypatch.setattr(heatmap, "cargar_centroides", lambda ciudad=None: centroides)
    monkeypatch.setattr(heatmap.webbrowser, "open", lambda _ruta: True)
    monkeypatch.chdir(tmp_path)

    heatmap.generar_mapa("Santa Cruz de la Sierra")

    contenido = (tmp_path / "data" / "mapa_precios.html").read_text(encoding="utf-8")
    assert "PRECIOS EN USD" in contenido
    assert "Captaciones extremas" in contenido
    assert "function enfocarExtremo" in contenido
    assert "CAP-VENTA" in contenido
    assert "precio_publicacion" in contenido
