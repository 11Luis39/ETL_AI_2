"""Orquestación reutilizable del pipeline ETL de INTRAMAX.

Este módulo no conoce FastAPI ni configura logging. La aplicación anfitriona
puede inyectar sus propios ``Engine`` de SQLAlchemy y decidir dónde guardar
los artefactos generados.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd
from sqlalchemy import Engine

from .cleaner import limpiar_datos
from .config import get_crm_engine, get_pg_engine
from .extractor import extraer_datos_crm
from .loader import cargar_datos, generar_dataset_entrenable
from .transformer import transformar_datos

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    """Opciones que cambian entre una ejecución y otra."""

    exportar_datasets: bool = False
    output_dir: Path = Path("data")


@dataclass(frozen=True)
class PipelineResult:
    """Resumen serializable de una ejecución del ETL."""

    extraidos: int
    limpios: int
    excluidos: int
    insertados: int
    actualizados: int
    duracion_segundos: float
    datasets_exportados: int = 0


def guardar_reporte_exclusiones(
    df_excluidos: pd.DataFrame,
    directorio: Path,
) -> None:
    """Guarda el detalle y el resumen de propiedades excluidas."""

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
    log.info("  -> Reportes de exclusión guardados en %s", directorio)


class ETLPipeline:
    """Servicio síncrono e independiente del framework web.

    Los engines son opcionales para conservar la ejecución por línea de
    comandos. En FastAPI deben inyectarse los engines administrados por la
    aplicación para reutilizar sus pools de conexiones.
    """

    def __init__(
        self,
        *,
        crm_engine: Engine | None = None,
        pg_engine: Engine | None = None,
    ) -> None:
        self._crm_engine = crm_engine
        self._pg_engine = pg_engine

    def run(self, options: PipelineOptions | None = None) -> PipelineResult:
        """Ejecuta extracción, limpieza, transformación y carga."""

        options = options or PipelineOptions()
        crm_engine = self._crm_engine or get_crm_engine()
        pg_engine = self._pg_engine or get_pg_engine()
        inicio = perf_counter()

        log.info("%s", "=" * 50)
        log.info("INTRAMAX ETL - Iniciando pipeline")
        log.info("%s", "=" * 50)

        log.info("PASO 1: Extrayendo datos del CRM...")
        df_raw = extraer_datos_crm(engine=crm_engine)
        if df_raw.empty:
            raise RuntimeError("El CRM no devolvió registros; se canceló la carga")
        log.info("  -> %s registros extraídos", len(df_raw))

        log.info("PASO 2: Limpiando datos...")
        df_clean, df_excluidos = limpiar_datos(df_raw)
        guardar_reporte_exclusiones(df_excluidos, options.output_dir)
        if df_clean.empty:
            raise RuntimeError("Todos los registros fueron excluidos durante la limpieza")
        log.info("  -> %s registros después de limpieza", len(df_clean))

        log.info("PASO 3: Transformando y calculando features...")
        df_final = transformar_datos(
            df_clean,
            pg_engine=pg_engine,
            crm_engine=crm_engine,
        )
        log.info("  -> %s registros listos para carga", len(df_final))

        log.info("PASO 4: Cargando a PostgreSQL...")
        insertados, actualizados = cargar_datos(df_final, engine=pg_engine)
        log.info("  -> %s insertados, %s actualizados", insertados, actualizados)

        datasets_exportados = 0
        if options.exportar_datasets:
            log.info("PASO 5: Generando datasets entrenables...")
            datasets_exportados = generar_dataset_entrenable(
                df_final,
                directorio=options.output_dir,
            )
            log.info("  -> %s operaciones exportadas", datasets_exportados)

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
            datasets_exportados=datasets_exportados,
        )


def run_pipeline(
    exportar_datasets: bool = False,
    *,
    crm_engine: Engine | None = None,
    pg_engine: Engine | None = None,
    output_dir: Path | str = Path("data"),
) -> PipelineResult:
    """Atajo compatible con el CLI y cómodo para una aplicación anfitriona."""

    pipeline = ETLPipeline(crm_engine=crm_engine, pg_engine=pg_engine)
    options = PipelineOptions(
        exportar_datasets=exportar_datasets,
        output_dir=Path(output_dir),
    )
    return pipeline.run(options)
