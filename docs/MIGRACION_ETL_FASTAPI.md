# Migración del ETL de INTRAMAX a un módulo FastAPI

## Objetivo

Incorporar el procesamiento analítico como un módulo aislado dentro de una
aplicación FastAPI existente, sin trasladar mapas, interfaces ni predicción de
precios. El código de datos debe seguir siendo independiente del framework;
FastAPI solamente inicia y consulta ejecuciones.

## Alcance mínimo

| Pieza | Migrar | Motivo |
|---|---:|---|
| `etl/pipeline.py` | Sí | Servicio y resultado de la ejecución |
| `etl/extractor.py` | Sí | Extracción desde el CRM MySQL |
| `etl/cleaner.py` | Sí | Reglas de calidad y normalización |
| `etl/transformer.py` | Sí | Features, fechas, precio/m² y zonas |
| `etl/loader.py` | Sí | Upsert en PostgreSQL y Parquet opcional |
| `etl/constants.py` | Sí | Ciudades y tipos de transacción |
| `etl/config.py` | Adaptar | Usar la configuración de la aplicación destino |
| `db/init.sql` y migraciones | Adaptar | Convertir las tablas analíticas a Alembic |
| `etl/clustering/setup_clusters.py` | Administrativa | Prepara `zona_clusters`; no debe ser un endpoint público |
| `etl/clustering/visualizar_*.py` | No | Son herramientas visuales, no ETL |
| `etl/modelo/` | No inicialmente | Entrenamiento y predicción son otro módulo |
| `etl/clustering/setup_avenidas.py` | No inicialmente | No participa en el flujo principal |

## Estructura sugerida en el proyecto destino

```text
app/
├── main.py
└── modules/
    └── intramax_etl/
        ├── __init__.py
        ├── pipeline.py
        ├── extractor.py
        ├── cleaner.py
        ├── transformer.py
        ├── loader.py
        ├── constants.py
        ├── dependencies.py     # engines y autorización del proyecto destino
        ├── schemas.py          # contratos HTTP, no DataFrames
        ├── router.py           # adaptador FastAPI pequeño
        └── jobs.py             # integración con la cola de trabajos
```

`pipeline.py` no debe importar FastAPI. Esto permite ejecutarlo desde una cola,
una tarea programada, pruebas o la línea de comandos sin levantar el servidor.

## Interfaz preparada

El núcleo actual ya acepta las conexiones administradas por la aplicación:

```python
from dataclasses import asdict
from pathlib import Path

from app.modules.intramax_etl import run_pipeline

resultado = run_pipeline(
    exportar_datasets=False,
    crm_engine=crm_sync_engine,
    pg_engine=analytics_sync_engine,
    output_dir=Path("var/intramax-etl") / job_id,
)

payload = asdict(resultado)
```

El resultado contiene:

- `extraidos`
- `limpios`
- `excluidos`
- `insertados`
- `actualizados`
- `duracion_segundos`
- `datasets_exportados`

## Adaptador FastAPI mínimo

Para una ejecución manual y controlada puede usarse un endpoint fino. Este
ejemplo evita bloquear el event loop, pero mantiene abierta la petición hasta
que termine el ETL:

```python
from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from .dependencies import get_analytics_engine, get_crm_engine
from .pipeline import run_pipeline

router = APIRouter(prefix="/etl", tags=["etl"])


@router.post("/run")
async def ejecutar_etl(
    crm_engine=Depends(get_crm_engine),
    analytics_engine=Depends(get_analytics_engine),
):
    resultado = await run_in_threadpool(
        run_pipeline,
        False,
        crm_engine=crm_engine,
        pg_engine=analytics_engine,
    )
    return asdict(resultado)
```

El router se registra desde la aplicación principal:

```python
from app.modules.intramax_etl.router import router as etl_router

app.include_router(etl_router, prefix="/api/v1")
```

## Ejecución recomendada en producción

El ETL consulta dos bases, procesa miles de filas con Pandas y calcula
distancias. No conviene ejecutarlo dentro del proceso web en producción.

La ruta recomendada es:

```text
POST /etl/jobs
      │
      ▼
crear job pendiente en base de datos
      │
      ▼
cola de trabajos → worker síncrono → ETLPipeline.run()
      │
      ▼
guardar resultado/error del job
      │
      ▼
GET /etl/jobs/{job_id}
```

El endpoint debe devolver `202 Accepted`. La tecnología de cola debe ser la
que ya utilice el proyecto (Celery, RQ, Dramatiq u otra); no se debe introducir
una segunda cola solo para este módulo.

## Conexiones de base de datos

- El ETL usa Pandas y SQLAlchemy síncronos.
- Si FastAPI usa `AsyncEngine`, el módulo ETL necesita engines síncronos
  independientes para MySQL y PostgreSQL.
- Los engines deben crearse una sola vez durante el ciclo de vida de la
  aplicación o del worker y reutilizar sus pools.
- No crear un engine por petición ni guardar credenciales dentro del router.
- Mantener `pool_pre_ping=True` y configurar límites de pool en el proyecto
  destino.

## Base analítica

Antes del primer procesamiento deben existir:

- `zona_clusters`
- `property_analytics`
- índices y restricciones asociadas

No se recomienda ejecutar `db/init.sql` desde una ruta HTTP. Las definiciones
necesarias deben convertirse en una revisión de Alembic dentro del proyecto
FastAPI. `zona_clusters` debe estar poblada antes del ETL; si está vacía, las
features geográficas quedan sin asignar o en cero.

`setup_clusters.py` todavía es interactivo y genera gráficas. Debe ejecutarse
como tarea administrativa durante la primera migración. Si más adelante se
automatiza, hay que separar el cálculo determinista de la selección manual de
`n_clusters`.

## Dependencias del módulo

El núcleo necesita las dependencias de `requirements.txt`:

- Pandas y NumPy
- scikit-learn
- SQLAlchemy
- PyMySQL
- psycopg2
- PyArrow cuando se exporten datasets

En el proyecto destino deben fusionarse con sus restricciones existentes. No
conviene instalar `requirements-model.txt` ni `requirements-geo.txt` para el
módulo ETL mínimo.

## Seguridad y concurrencia

- Proteger cualquier ruta ETL con el sistema de autorización administrativa
  del proyecto existente.
- Impedir dos ejecuciones simultáneas sobre el mismo destino.
- Usar un directorio de salida distinto por `job_id` para no sobrescribir
  reportes.
- No devolver excepciones SQL ni credenciales al cliente; guardar el detalle
  en logs y exponer un mensaje controlado.
- El loader usa upsert por `id_propiedad`, por lo que repetir una ejecución es
  mayormente idempotente.

## Orden de migración

1. Crear el paquete `app.modules.intramax_etl`.
2. Adaptar imports relativos y la configuración del proyecto destino.
3. Crear la migración Alembic de las tablas analíticas.
4. Copiar o generar los centroides de `zona_clusters`.
5. Inyectar los dos engines síncronos en `ETLPipeline`.
6. Ejecutar las pruebas unitarias sin bases reales.
7. Hacer una corrida de ensayo contra una base analítica temporal.
8. Comparar conteos y muestras con el sistema actual.
9. Conectar el pipeline a la cola de trabajos y finalmente al router.

## Criterios de aceptación

- Mismos registros extraídos y excluidos que el proyecto original.
- Mismos valores de precios USD, superficies y features para una muestra fija.
- Upsert repetible sin duplicar `id_propiedad`.
- Un fallo del ETL no detiene el servidor FastAPI.
- No hay secretos, configuración global ni rutas de archivos codificadas en el
  módulo.
- Las visualizaciones y modelos quedan fuera del despliegue ETL.
