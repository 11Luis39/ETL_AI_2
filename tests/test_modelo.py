from pathlib import Path

import pandas as pd
import pytest

from etl.modelo.entrenar_precio import cargar_dataset
from etl.modelo.predecir import predecir


def test_cargar_dataset_usa_el_archivo_generado_por_el_etl(tmp_path: Path):
    ruta = tmp_path / "dataset_ventas.parquet"
    pd.DataFrame(
        {
            "tipo_propiedad": ["Terreno", "Terreno", "Casa"],
            "m2_terreno": [500, 2_000, 300],
            "precio_venta": [10_000, 20_000, 30_000],
            "tiempo_en_mercado": [30, 45, 20],
        }
    ).to_parquet(ruta, index=False)

    df = cargar_dataset(ruta)

    assert df["tipo_propiedad"].tolist() == [
        "Terreno_urbano",
        "Terreno_rural",
        "Casa",
    ]


def test_cargar_dataset_explica_como_generar_un_archivo_faltante(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="--exportar-datasets"):
        cargar_dataset(tmp_path / "inexistente.parquet")


def test_predecir_exige_ciudad_antes_de_consultar_la_base():
    resultado = predecir({}, {"tipo_propiedad": "Casa"})

    assert resultado == {"error": "La ciudad es obligatoria para asignar una zona correcta"}
