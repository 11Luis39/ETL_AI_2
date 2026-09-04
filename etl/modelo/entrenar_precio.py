import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Features por tipo de propiedad
# ------------------------------------------------------------
FEATURES_BASE = [
    "cluster_zona",
    "m2_construidos",
    "dormitorios",
    "banos",
    "antiguedad",
    "tiempo_en_mercado",
    "mes_publicacion",
    "ratio_activas_vendidas_zona",
    "diferencia_vs_promedio_zona",
]

FEATURES_POR_TIPO = {
    "Casa": FEATURES_BASE + ["estacionamientos"] + ["m2_terreno"],
    "Departamento": FEATURES_BASE + ["estacionamientos"],
    "Terreno_urbano": [
        "cluster_zona",
        "m2_terreno",
        "mes_publicacion",
        "ratio_activas_vendidas_zona",
        "diferencia_vs_promedio_zona",
    ],
    "Terreno_rural": [
        "cluster_zona",
        "m2_terreno",
        "mes_publicacion",
        "ratio_activas_vendidas_zona",
    ],
}

TARGET_POR_TIPO = {
    "Casa": "precio_venta",
    "Departamento": "precio_venta",
    "Terreno_urbano": "precio_m2",
    "Terreno_rural": "precio_m2",
}

TIPOS_MVP = ["Casa", "Departamento", "Terreno_urbano", "Terreno_rural"]


# ------------------------------------------------------------
# Cargar dataset
# ------------------------------------------------------------
def cargar_dataset(ruta: Path = Path("data/dataset_ventas.parquet")) -> pd.DataFrame:
    """Carga el dataset de ventas generado por ``etl.main --exportar-datasets``."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecutá primero: python -m etl.main --exportar-datasets"
        )
    df = pd.read_parquet(ruta)
    log.info(f"Dataset cargado: {len(df)} registros")

    df = df[df["precio_venta"].notna()].copy()
    df = df[df["tiempo_en_mercado"].notna()].copy()

    # Segmentar terrenos sin recorrer el DataFrame fila por fila.
    es_terreno = df["tipo_propiedad"].eq("Terreno")
    superficie = pd.to_numeric(df["m2_terreno"], errors="coerce").fillna(0)
    df.loc[es_terreno, "tipo_propiedad"] = np.where(
        superficie.loc[es_terreno].le(1_000),
        "Terreno_urbano",
        "Terreno_rural",
    )
    df = df[df["tipo_propiedad"].isin(TIPOS_MVP)].copy()

    log.info(f"Dataset filtrado: {len(df)} registros para entrenamiento")

    # Mostrar distribución
    for tipo in TIPOS_MVP:
        n = len(df[df["tipo_propiedad"] == tipo])
        log.info(f"  {tipo}: {n} registros")

    return df


# ------------------------------------------------------------
# Entrenar modelo por tipo
# ------------------------------------------------------------
def entrenar_modelo_tipo(df: pd.DataFrame, tipo: str):
    log.info(f"\n{'=' * 50}")
    log.info(f"Entrenando modelo: {tipo}")
    log.info(f"{'=' * 50}")

    df_tipo = df[df["tipo_propiedad"] == tipo].copy()
    features = FEATURES_POR_TIPO[tipo]

    # Verificar columnas disponibles
    features = [f for f in features if f in df_tipo.columns]
    log.info(f"Features usados: {features}")
    log.info(f"Registros: {len(df_tipo)}")

    target = TARGET_POR_TIPO.get(tipo, "precio_venta")
    log.info(f"Target: {target}")

    X = df_tipo[features].copy()
    y = df_tipo[target].copy()

    # Filtrar filas donde el target es nulo
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    log.info(f"Registros con target válido: {len(X)}")

    # Convertir columnas a numérico y rellenar nulos con mediana
    medianas = {}
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        mediana = X[col].median()
        if pd.isna(mediana):
            mediana = 0.0
        medianas[col] = float(mediana)
        if X[col].isna().any():
            X[col] = X[col].fillna(mediana)
            log.info(f"  Nulos en '{col}' rellenados con mediana: {mediana:.2f}")

    # Clampear features con outliers extremos
    CLAMP_FEATURES = {
        "diferencia_vs_promedio_zona": (-2.0, 2.0),
        "ratio_activas_vendidas_zona": (0.0, 10.0),
    }
    for col, (min_val, max_val) in CLAMP_FEATURES.items():
        if col in X.columns:
            antes_min = X[col].min()
            antes_max = X[col].max()
            X[col] = X[col].clip(lower=min_val, upper=max_val)
            if antes_min < min_val or antes_max > max_val:
                log.info(
                    f"  Clamp '{col}': [{antes_min:.2f}, {antes_max:.2f}] → [{min_val}, {max_val}]"
                )

    # Split train/test 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Modelo XGBoost
    modelo = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Métricas
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    log.info(f"  MAE:  {mae:,.2f}")
    log.info(f"  RMSE: {rmse:,.2f}")
    log.info(f"  MAPE: {mape:.2f}%")

    if mape <= 12:
        log.info("  ✅ Meta cumplida (MAPE ≤ 12%)")
    else:
        log.info("  ⚠️  Meta no cumplida (MAPE > 12%) — se puede mejorar con más datos")

    # Importancia de features
    log.info("  Importancia de features:")
    importancias = pd.Series(modelo.feature_importances_, index=features)
    for feat, imp in importancias.sort_values(ascending=False).items():
        log.info(f"    {feat}: {imp:.4f}")

    return (
        modelo,
        features,
        medianas,
        {
            "tipo": tipo,
            "registros": len(df_tipo),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        },
    )


# ------------------------------------------------------------
# Guardar modelos
# ------------------------------------------------------------
def guardar_modelo(modelo, features: list, medianas: dict, tipo: str):
    os.makedirs("data/modelos", exist_ok=True)
    nombre = tipo.lower().replace(" ", "_")
    ruta = f"data/modelos/precio_{nombre}.joblib"

    joblib.dump(
        {
            "modelo": modelo,
            "features": features,
            "medianas": medianas,
            "tipo": tipo,
            "target": TARGET_POR_TIPO.get(tipo, "precio_venta"),
            "fecha": datetime.now().isoformat(),
            "version": "1.0.0",
        },
        ruta,
    )

    log.info(f"  Modelo guardado: {ruta}")
    return ruta


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    log.info("INTRAMAX — Entrenamiento Modelo de Precio")

    df = cargar_dataset()

    resultados = []
    for tipo in TIPOS_MVP:
        df_tipo = df[df["tipo_propiedad"] == tipo]
        if len(df_tipo) < 50:
            log.warning(f"  {tipo}: solo {len(df_tipo)} registros, saltando...")
            continue

        modelo, features, medianas, metricas = entrenar_modelo_tipo(df, tipo)
        guardar_modelo(modelo, features, medianas, tipo)
        resultados.append(metricas)

    # Resumen final
    log.info(f"\n{'=' * 50}")
    log.info("RESUMEN FINAL")
    log.info(f"{'=' * 50}")
    for r in resultados:
        log.info(
            f"  {r['tipo']:15} | "
            f"Registros: {r['registros']:4} | "
            f"MAE: {r['mae']:>10,.0f} | "
            f"MAPE: {r['mape']:>5.1f}%"
        )
