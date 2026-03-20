-- ============================================================
-- INTRAMAX IPPE — Schema Analítico
-- Versión 3.0 — Multiciudad + Venta + Alquiler
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------
-- TABLA: zona_clusters
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zona_clusters (
    id                  SERIAL,
    cluster_id          SMALLINT        NOT NULL,
    ciudad              VARCHAR(100)    NOT NULL DEFAULT 'Santa Cruz de la Sierra',
    pais                VARCHAR(50)     NOT NULL DEFAULT 'Bolivia',
    centroide_lat       NUMERIC(10,7)   NOT NULL,
    centroide_lng       NUMERIC(10,7)   NOT NULL,
    total_propiedades   INT             DEFAULT 0,
    precio_m2_promedio  NUMERIC(10,2),
    n_clusters_ciudad   SMALLINT,
    fecha_generacion    TIMESTAMP       DEFAULT NOW(),

    PRIMARY KEY (cluster_id, ciudad)
);

CREATE INDEX IF NOT EXISTS idx_zc_ciudad  ON zona_clusters(ciudad);
CREATE INDEX IF NOT EXISTS idx_zc_cluster ON zona_clusters(cluster_id);

-- ------------------------------------------------------------
-- TABLA: property_analytics
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS property_analytics (
    id                          SERIAL PRIMARY KEY,
    id_propiedad                VARCHAR(64)     NOT NULL UNIQUE,
    mlsid                       VARCHAR(64),

    -- Tipo
    tipo_propiedad              VARCHAR(30)     NOT NULL,
    subtipo_original            VARCHAR(100),
    categoria_propiedad         VARCHAR(100),
    estado_propiedad            VARCHAR(100),
    segmento                    VARCHAR(50), 

    -- Tipo de transacción
    tipo_transaccion            VARCHAR(20)     NOT NULL DEFAULT 'Venta',

    -- Ubicación
    latitude                    NUMERIC(10,7),
    longitude                   NUMERIC(10,7),
    cluster_zona                SMALLINT,
    ciudad                      VARCHAR(100)    NOT NULL DEFAULT 'Santa Cruz de la Sierra',
    pais                        VARCHAR(50)     NOT NULL DEFAULT 'Bolivia',

    -- Características físicas
    m2_construidos              NUMERIC(10,2),
    m2_terreno                  NUMERIC(10,2),
    dormitorios                 SMALLINT,
    banos                       SMALLINT,
    estacionamientos            SMALLINT,
    antiguedad                  SMALLINT,

    -- Precios
    precio_publicacion          NUMERIC(15,2),
    precio_venta                NUMERIC(15,2),      -- solo Venta
    precio_alquiler_mes         NUMERIC(15,2),      -- solo Alquiler (BOB/mes)
    precio_m2                   NUMERIC(10,2),

    -- Métricas de mercado
    tiempo_en_mercado           SMALLINT,
    numero_reducciones          SMALLINT        DEFAULT 0,
    diferencia_vs_promedio_zona NUMERIC(8,4),
    ratio_activas_vendidas_zona NUMERIC(8,4),

    -- Fechas
    mes_publicacion             SMALLINT,
    anio_publicacion            SMALLINT,
    fecha_venta                 DATE,

    -- Estado
    status                      VARCHAR(50),
    transaction_type            VARCHAR(20),

    -- Control ETL
    fecha_carga                 TIMESTAMP       DEFAULT NOW(),
    fecha_actualizacion         TIMESTAMP       DEFAULT NOW(),

    CONSTRAINT fk_cluster_ciudad
        FOREIGN KEY (cluster_zona, ciudad)
        REFERENCES zona_clusters(cluster_id, ciudad)
        ON DELETE SET NULL
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX IF NOT EXISTS idx_pa_cluster      ON property_analytics(cluster_zona);
CREATE INDEX IF NOT EXISTS idx_pa_tipo         ON property_analytics(tipo_propiedad);
CREATE INDEX IF NOT EXISTS idx_pa_transaccion  ON property_analytics(tipo_transaccion);
CREATE INDEX IF NOT EXISTS idx_pa_status       ON property_analytics(status);
CREATE INDEX IF NOT EXISTS idx_pa_fecha_venta  ON property_analytics(fecha_venta);
CREATE INDEX IF NOT EXISTS idx_pa_coords       ON property_analytics(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_pa_ciudad       ON property_analytics(ciudad);

-- ------------------------------------------------------------
-- TABLA: prediction_logs
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_logs (
    id                          SERIAL PRIMARY KEY,
    id_prediccion               UUID            DEFAULT uuid_generate_v4(),
    id_propiedad                VARCHAR(64),

    ciudad                      VARCHAR(100),
    pais                        VARCHAR(50)     DEFAULT 'Bolivia',
    tipo_transaccion            VARCHAR(20)     DEFAULT 'Venta',

    precio_input                NUMERIC(15,2)   NOT NULL,
    tipo_propiedad_input        VARCHAR(30),
    m2_input                    NUMERIC(10,2),
    cluster_zona_input          SMALLINT,

    precio_predicho             NUMERIC(15,2),
    rango_min                   NUMERIC(15,2),
    rango_max                   NUMERIC(15,2),

    alquiler_predicho_mes       NUMERIC(15,2),
    alquiler_rango_min          NUMERIC(15,2),
    alquiler_rango_max          NUMERIC(15,2),

    confianza_modelo            NUMERIC(5,4),
    probabilidad_venta_60       NUMERIC(5,4),
    precio_optimo_recomendado   NUMERIC(15,2),
    mejora_probabilidad         NUMERIC(5,4),

    resultado_real              NUMERIC(15,2),
    modelo_version              VARCHAR(20),

    fecha_prediccion            TIMESTAMP       DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pl_propiedad   ON prediction_logs(id_propiedad);
CREATE INDEX IF NOT EXISTS idx_pl_fecha       ON prediction_logs(fecha_prediccion);
CREATE INDEX IF NOT EXISTS idx_pl_ciudad      ON prediction_logs(ciudad);
CREATE INDEX IF NOT EXISTS idx_pl_transaccion ON prediction_logs(tipo_transaccion);