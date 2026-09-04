"""Punto de entrada del pipeline ETL de INTRAMAX."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import PipelineResult as PipelineResult
from .pipeline import guardar_reporte_exclusiones as guardar_reporte_exclusiones
from .pipeline import run_pipeline

log = logging.getLogger(__name__)


def configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline analítico INTRAMAX")
    parser.add_argument(
        "--exportar-datasets",
        action="store_true",
        help="genera archivos Parquet para entrenamiento al finalizar",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="directorio para reportes y datasets (por defecto: data)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configurar_logging()
    args = parse_args(argv)
    try:
        run_pipeline(
            exportar_datasets=args.exportar_datasets,
            output_dir=args.output_dir,
        )
    except Exception:
        log.exception("ETL falló")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
