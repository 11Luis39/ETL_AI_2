"""Punto de entrada del pipeline ETL de INTRAMAX."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from .cleaner import limpiar_datos
from .extractor import extraer_datos_crm
from .loader import cargar_datos, generar_dataset_entrenable
from .transformer import transformar_datos

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    extraidos: int
    limpios: int
    excluidos: int
    insertados: int
    actualizados: int
    duracion_segundos: float


def configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def guardar_reporte_exclusiones(
    df_excluidos: pd.DataFrame,
    directorio: Path = Path("data"),
) -> None:
    """Guarda detalle por propiedad y un resumen por motivo."""

    if df_excluidos.empty:
        return

    directorio.mkdir(parents=True, exist_ok=True)
    contexto = [
        "mlsid",
        "subtipo_original",
        "tipo_propiedad",
        "segmento",
        "tipo_transaccion",
    ]
    agregaciones = {columna: "first" for columna in contexto}
    agregaciones["motivo"] = lambda valores: ", ".join(sorted(set(valores)))
    detalle = (
        df_excluidos.groupby("id_propiedad", dropna=False, as_index=False)
        .agg(agregaciones)
        .rename(columns={"motivo": "motivos"})
    )
    detalle.to_csv(directorio / "captaciones_excluidas.csv", index=False)

    resumen = (
        df_excluidos.drop_duplicates(subset=["id_propiedad", "motivo"])["motivo"]
        .value_counts()
        .rename_axis("motivo")
        .reset_index(name="total")
    )
    resumen.to_csv(directorio / "captaciones_excluidas_resumen.csv", index=False)
    log.info("  → Reportes de exclusión guardados en %s", directorio)


def run_pipeline(exportar_datasets: bool = False) -> PipelineResult:
    """Ejecuta extracción, limpieza, transformación y carga."""

    inicio = perf_counter()
    log.info("%s", "=" * 50)
    log.info("INTRAMAX ETL — Iniciando pipeline")
    log.info("%s", "=" * 50)

    log.info("PASO 1: Extrayendo datos del CRM...")
    df_raw = extraer_datos_crm()
    if df_raw.empty:
        raise RuntimeError("El CRM no devolvió registros; se canceló la carga")
    log.info("  → %s registros extraídos", len(df_raw))

    log.info("PASO 2: Limpiando datos...")
    df_clean, df_excluidos = limpiar_datos(df_raw)
    guardar_reporte_exclusiones(df_excluidos)
    if df_clean.empty:
        raise RuntimeError("Todos los registros fueron excluidos durante la limpieza")
    log.info("  → %s registros después de limpieza", len(df_clean))

    log.info("PASO 3: Transformando y calculando features...")
    df_final = transformar_datos(df_clean)
    log.info("  → %s registros listos para carga", len(df_final))

    log.info("PASO 4: Cargando a PostgreSQL...")
    insertados, actualizados = cargar_datos(df_final)
    log.info("  → %s insertados, %s actualizados", insertados, actualizados)

    if exportar_datasets:
        log.info("PASO 5: Generando datasets entrenables...")
        total_entrenable = generar_dataset_entrenable(df_final)
        log.info("  → %s operaciones exportadas", total_entrenable)

    duracion = perf_counter() - inicio
    log.info("%s", "=" * 50)
    log.info("ETL completado exitosamente en %.2f segundos", duracion)
    log.info("%s", "=" * 50)
    return PipelineResult(
        extraidos=len(df_raw),
        limpios=len(df_clean),
        excluidos=df_excluidos["id_propiedad"].nunique(dropna=False),
        insertados=insertados,
        actualizados=actualizados,
        duracion_segundos=duracion,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline analítico INTRAMAX")
    parser.add_argument(
        "--exportar-datasets",
        action="store_true",
        help="genera archivos Parquet para entrenamiento al finalizar",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configurar_logging()
    args = parse_args(argv)
    try:
        run_pipeline(exportar_datasets=args.exportar_datasets)
    except Exception:
        log.exception("ETL falló")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
