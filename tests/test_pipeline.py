from pathlib import Path

import pandas as pd

from etl import pipeline


def test_pipeline_reutiliza_engines_y_directorio_inyectados(monkeypatch, tmp_path: Path):
    crm_engine = object()
    pg_engine = object()
    df_raw = pd.DataFrame({"id_propiedad": ["1", "2"]})
    df_clean = df_raw.copy()
    df_final = df_raw.assign(tipo_transaccion=["Venta", "Alquiler"])
    df_excluidos = pd.DataFrame(columns=["id_propiedad"])
    llamadas: dict[str, object] = {}

    def extraer(*, engine):
        llamadas["crm_extractor"] = engine
        return df_raw

    def limpiar(df):
        assert df is df_raw
        return df_clean, df_excluidos

    def transformar(df, *, pg_engine, crm_engine):
        assert df is df_clean
        llamadas["crm_transformer"] = crm_engine
        llamadas["pg_transformer"] = pg_engine
        return df_final

    def cargar(df, *, engine):
        assert df is df_final
        llamadas["pg_loader"] = engine
        return 1, 1

    def exportar(df, *, directorio):
        assert df is df_final
        llamadas["directorio"] = directorio
        return 2

    monkeypatch.setattr(pipeline, "extraer_datos_crm", extraer)
    monkeypatch.setattr(pipeline, "limpiar_datos", limpiar)
    monkeypatch.setattr(pipeline, "transformar_datos", transformar)
    monkeypatch.setattr(pipeline, "cargar_datos", cargar)
    monkeypatch.setattr(pipeline, "generar_dataset_entrenable", exportar)

    resultado = pipeline.run_pipeline(
        exportar_datasets=True,
        crm_engine=crm_engine,
        pg_engine=pg_engine,
        output_dir=tmp_path,
    )

    assert llamadas == {
        "crm_extractor": crm_engine,
        "crm_transformer": crm_engine,
        "pg_transformer": pg_engine,
        "pg_loader": pg_engine,
        "directorio": tmp_path,
    }
    assert resultado.extraidos == 2
    assert resultado.limpios == 2
    assert resultado.insertados == 1
    assert resultado.actualizados == 1
    assert resultado.datasets_exportados == 2
