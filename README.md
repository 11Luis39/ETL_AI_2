# INTRAMAX IA

Pipeline analítico para extraer operaciones inmobiliarias del CRM, limpiarlas,
calcular variables de zona y cargarlas en PostgreSQL. El repositorio también
incluye utilidades de clustering, mapas y modelos de precio.

## Convención monetaria

Todos los importes se almacenan y procesan en dólares estadounidenses (USD):
precio de publicación, precio de cierre de venta, alquiler mensual, precio por
m², datasets y predicciones. El pipeline no realiza conversiones de moneda.

## Flujo principal

```text
CRM MySQL
   │
   ▼
extractor → cleaner → transformer → loader → PostgreSQL
               │                         │
               └─ reportes CSV           └─ datasets Parquet opcionales
```

## Estructura

- `etl/config.py`: variables de entorno y conexiones compartidas.
- `etl/constants.py`: ciudades y tipos de transacción.
- `etl/extractor.py`: consultas al CRM.
- `etl/cleaner.py`: validaciones, normalización y reporte de exclusiones.
- `etl/transformer.py`: fechas, precio por m², clusters y features de zona.
- `etl/loader.py`: upsert por lotes y exportación de datasets.
- `etl/pipeline.py`: servicio reutilizable e inyectable del pipeline.
- `etl/main.py`: adaptador de línea de comandos.
- `etl/clustering/`: creación de zonas y visualizaciones geográficas.
- `etl/modelo/`: entrenamiento y predicción de precios.
- `db/init.sql`: esquema analítico de PostgreSQL.
- `tests/`: pruebas del núcleo ETL que no requieren bases de datos.

## Preparación local

1. Crear y activar un entorno virtual de Python 3.11.
2. Instalar el núcleo:

   ```powershell
   pip install -r requirements.txt
   ```

3. Para mapas y procesamiento geográfico:

   ```powershell
   pip install -r requirements-geo.txt
   ```

4. Para entrenar o ejecutar los modelos de precio:

   ```powershell
   pip install -r requirements-model.txt
   ```

5. Copiar `.env.example` a `.env` y completar las credenciales reales. `.env`
   está ignorado por Git y no se copia dentro de la imagen Docker.

## Ejecución

Desde la raíz del proyecto:

```powershell
python -m etl.main
```

Para generar además los Parquet de entrenamiento:

```powershell
python -m etl.main --exportar-datasets
```

Con Docker:

```powershell
docker compose up --build
```

PostgreSQL queda publicado en el puerto `5433`; dentro de Docker el ETL usa el
nombre de servicio `postgres_analitico` automáticamente.

Si la base ya existía antes de esta reorganización, aplicar una vez la
migración de la clave foránea de clusters:

```powershell
Get-Content db/migrations/001_fix_cluster_foreign_key.sql -Raw |
  docker compose exec -T postgres_analitico sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

## Calidad

Instalar las herramientas de desarrollo y ejecutar:

```powershell
pip install -r requirements-dev.txt
pytest
ruff check etl tests
```

Los archivos generados se guardan bajo `data/`, que está ignorado por Git.

## Migración a FastAPI

El núcleo ETL no depende de FastAPI y acepta engines SQLAlchemy inyectados por
la aplicación anfitriona. La selección de archivos, arquitectura objetivo,
adaptador HTTP y estrategia de trabajos están documentados en
[`docs/MIGRACION_ETL_FASTAPI.md`](docs/MIGRACION_ETL_FASTAPI.md).

## Herramientas auxiliares

Ejecutá también estos scripts como módulos, siempre desde la raíz:

```powershell
# Crear clusters de ciudades
python -m etl.clustering.setup_clusters

# Preparar y visualizar la red vial
python -m etl.clustering.setup_avenidas
python -m etl.clustering.visualizar_avenidas

# Mapas de clusters y precios
python -m etl.clustering.visualizar_clusters
python -m etl.clustering.visualizar_heatmap

# Entrenar y usar modelos
python -m etl.modelo.entrenar_precio
python -m etl.modelo.predecir
```
