import logging
import os
import sys
from datetime import datetime

from extractor import extraer_datos_crm
from cleaner import limpiar_datos
from transformer import transformar_datos
from loader import cargar_datos, generar_dataset_entrenable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def run_pipeline():
    inicio = datetime.now()
    log.info("=" * 50)
    log.info("INTRAMAX ETL — Iniciando pipeline")
    log.info("=" * 50)

    try:
        # Paso 1 — Extracción
        log.info("PASO 1: Extrayendo datos del CRM...")
        df_raw = extraer_datos_crm()
        log.info(f"  → {len(df_raw)} registros extraídos")

        # Paso 2 — Limpieza
        log.info("PASO 2: Limpiando datos...")
        df_clean, df_excluidos = limpiar_datos(df_raw)
        if not df_excluidos.empty:
            os.makedirs("data", exist_ok=True)

            # Detalle — una fila por propiedad con todos sus motivos
            detalle = (
                df_excluidos
                .groupby(["id_propiedad", "mlsid", "subtipo_original",
                          "tipo_propiedad", "tipo_transaccion"])["motivo"]
                .apply(lambda motivos: ", ".join(sorted(set(motivos))))
                .reset_index()
                .rename(columns={"motivo": "motivos"})
            )
            detalle.to_csv("data/captaciones_excluidas.csv", index=False)
            log.info(f"  → Detalle guardado en data/captaciones_excluidas.csv")

            # Resumen — propiedades únicas por motivo
            resumen = (
                df_excluidos
                .drop_duplicates(subset=["id_propiedad", "motivo"])["motivo"]
                .value_counts()
                .reset_index()
            )
            resumen.columns = ["motivo", "total"]
            resumen.to_csv("data/captaciones_excluidas_resumen.csv", index=False)
            log.info(f"  → Resumen guardado en data/captaciones_excluidas_resumen.csv")
        log.info(f"  → {len(df_clean)} registros después de limpieza")

        # Paso 3 — Transformación + features derivadas
        log.info("PASO 3: Transformando y calculando features...")
        df_final = transformar_datos(df_clean)
        log.info(f"  → {len(df_final)} registros listos para carga")

        # Paso 4 — Carga a PostgreSQL
        log.info("PASO 4: Cargando a PostgreSQL...")
        insertados, actualizados = cargar_datos(df_final)
        log.info(f"  → {insertados} insertados, {actualizados} actualizados")

        # Paso 5 — Dataset entrenable
        log.info("PASO 5: Generando dataset entrenable...")
        total_entrenable = generar_dataset_entrenable(df_final)
        log.info(f"  → {total_entrenable} propiedades vendidas exportadas")

        duracion = (datetime.now() - inicio).seconds
        log.info("=" * 50)
        log.info(f"ETL completado exitosamente en {duracion} segundos")
        log.info("=" * 50)

    except Exception as e:
        log.error(f"ETL falló: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()