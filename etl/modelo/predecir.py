from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sqlalchemy import Engine, text

from etl.config import get_pg_engine


# ------------------------------------------------------------
# Cargar modelos
# ------------------------------------------------------------
def cargar_modelos(ruta_base: Path = Path("data/modelos")):
    if not ruta_base.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_base}. Entrená los modelos antes de predecir."
        )
    modelos = {}
    for ruta in sorted(ruta_base.glob("*.joblib")):
        if ruta.is_file():
            data = joblib.load(ruta)
            modelos[data["tipo"]] = data
            print(f"✓ Modelo cargado: {data['tipo']} (v{data['version']})")
    return modelos


# ------------------------------------------------------------
# Asignar cluster desde coordenadas
# ------------------------------------------------------------
def obtener_cluster(
    lat: float,
    lng: float,
    ciudad: str,
    engine: Engine | None = None,
) -> int:
    """Busca el centroide más cercano dentro de la ciudad indicada."""

    engine = engine or get_pg_engine()

    with engine.connect() as conn:
        centroides = pd.read_sql(
            text("""
                SELECT cluster_id, centroide_lat, centroide_lng
                FROM zona_clusters
                WHERE ciudad = :ciudad
            """),
            conn,
            params={"ciudad": ciudad},
        )

    if centroides.empty:
        raise ValueError(f"No hay clusters configurados para {ciudad}")

    coords = np.array([[lat, lng]])
    cents = centroides[["centroide_lat", "centroide_lng"]].values
    distancias = pairwise_distances(coords, cents, metric="euclidean")
    idx = distancias.argmin()
    return int(centroides["cluster_id"].values[idx])


# ------------------------------------------------------------
# Obtener features de zona desde PostgreSQL
# ------------------------------------------------------------
def obtener_features_zona(
    cluster_zona: int,
    ciudad: str,
    engine: Engine | None = None,
) -> dict:
    engine = engine or get_pg_engine()

    query = text("""
        SELECT
            AVG(ratio_activas_vendidas_zona) as ratio,
            AVG(diferencia_vs_promedio_zona) as diferencia
        FROM property_analytics
        WHERE cluster_zona = :cluster
          AND ciudad = :ciudad
    """)

    with engine.connect() as conn:
        result = conn.execute(
            query,
            {"cluster": cluster_zona, "ciudad": ciudad},
        ).fetchone()

    return {
        "ratio_activas_vendidas_zona": float(result[0]) if result[0] else 0.0,
        "diferencia_vs_promedio_zona": float(result[1]) if result[1] else 0.0,
    }


# ------------------------------------------------------------
# Predecir
# ------------------------------------------------------------
def predecir(
    modelos: dict,
    datos: dict,
    engine: Engine | None = None,
) -> dict:
    tipo = datos["tipo_propiedad"]
    ciudad = str(datos.get("ciudad", "")).strip()
    if not ciudad:
        return {"error": "La ciudad es obligatoria para asignar una zona correcta"}

    # Mapear tipo a modelo correcto
    if tipo == "Terreno":
        m2_terreno = datos.get("m2_terreno", 0) or 0
        tipo_modelo = "Terreno_urbano" if m2_terreno <= 1000 else "Terreno_rural"
    else:
        tipo_modelo = tipo

    if tipo_modelo not in modelos:
        return {"error": f"No hay modelo disponible para tipo: {tipo_modelo}"}

    modelo_data = modelos[tipo_modelo]
    modelo = modelo_data["modelo"]
    features = modelo_data["features"]
    target = modelo_data["target"]
    medianas = modelo_data.get("medianas", {})

    # Obtener cluster y features de zona
    engine = engine or get_pg_engine()
    try:
        cluster = obtener_cluster(datos["latitude"], datos["longitude"], ciudad, engine=engine)
        zona = obtener_features_zona(cluster, ciudad, engine=engine)
    except ValueError as exc:
        return {"error": str(exc)}

    # Construir fila de features
    fila = {
        "cluster_zona": cluster,
        "m2_construidos": datos.get("m2_construidos", 0),
        "m2_terreno": datos.get("m2_terreno", 0),
        "dormitorios": datos.get("dormitorios", 0),
        "banos": datos.get("banos", 0),
        "antiguedad": datos.get("antiguedad", 0),
        "tiempo_en_mercado": datos.get("tiempo_en_mercado", 30),
        "mes_publicacion": datos.get("mes_publicacion", 3),
        "estacionamientos": datos.get("estacionamientos", 0),
        "ratio_activas_vendidas_zona": min(zona["ratio_activas_vendidas_zona"], 10.0),
        "diferencia_vs_promedio_zona": max(-2.0, min(zona["diferencia_vs_promedio_zona"], 2.0)),
    }

    X = pd.DataFrame([{feature: fila.get(feature) for feature in features}])

    # Aplicar la misma imputación usada durante el entrenamiento.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(float(medianas.get(col, 0)))

    prediccion = float(modelo.predict(X)[0])

    # Si el modelo predice precio_m2, calcular precio total
    if target == "precio_m2":
        m2 = datos.get("m2_terreno", 0) or datos.get("m2_construidos", 0)
        precio_total = prediccion * m2
        return {
            "tipo_modelo": tipo_modelo,
            "cluster_zona": cluster,
            "precio_m2": round(prediccion, 2),
            "m2": m2,
            "precio_total": round(precio_total, 2),
            "rango_min": round(precio_total * 0.88, 2),
            "rango_max": round(precio_total * 1.12, 2),
            "moneda": "USD",
        }
    else:
        return {
            "tipo_modelo": tipo_modelo,
            "cluster_zona": cluster,
            "precio_total": round(prediccion, 2),
            "rango_min": round(prediccion * 0.88, 2),
            "rango_max": round(prediccion * 1.12, 2),
            "moneda": "USD",
        }


# ------------------------------------------------------------
# Main — prueba interactiva
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("INTRAMAX — Motor de Predicción de Precio")
    print("=" * 50)

    modelos = cargar_modelos()

    print("\n¿Qué tipo de propiedad querés predecir?")
    print("1. Casa")
    print("2. Departamento")
    print("3. Terreno")
    opcion = input("Elegí (1/2/3): ").strip()

    tipo_map = {"1": "Casa", "2": "Departamento", "3": "Terreno"}
    tipo = tipo_map.get(opcion, "Departamento")

    print(f"\n--- Datos de la {tipo} ---")

    datos = {"tipo_propiedad": tipo}
    datos["ciudad"] = input("Ciudad (ej: Santa Cruz de la Sierra): ").strip()

    if tipo == "Terreno":
        datos["m2_terreno"] = float(input("Superficie del terreno (m2): "))
        datos["latitude"] = float(input("Latitud (ej: -17.783): "))
        datos["longitude"] = float(input("Longitud (ej: -63.182): "))
        datos["antiguedad"] = int(input("Antigüedad en años (0 si es nuevo): "))
        datos["mes_publicacion"] = 3
    else:
        datos["m2_construidos"] = float(input("Metros construidos (m2): "))
        datos["dormitorios"] = int(input("Dormitorios: "))
        datos["banos"] = int(input("Baños: "))
        datos["estacionamientos"] = int(input("Estacionamientos: "))
        datos["antiguedad"] = int(input("Antigüedad en años (0 si es nuevo): "))
        datos["latitude"] = float(input("Latitud (ej: -17.783): "))
        datos["longitude"] = float(input("Longitud (ej: -63.182): "))
        datos["mes_publicacion"] = 3

    print("\n⏳ Calculando...")
    resultado = predecir(modelos, datos)

    print("\n" + "=" * 50)
    print("RESULTADO")
    print("=" * 50)

    if "error" in resultado:
        print(f"❌ {resultado['error']}")
    else:
        print(f"  Tipo modelo:   {resultado['tipo_modelo']}")
        print(f"  Zona (cluster): {resultado['cluster_zona']}")
        if "precio_m2" in resultado:
            print(f"  Precio/m2:     {resultado['precio_m2']:,.2f} USD")
            print(f"  Superficie:    {resultado['m2']:,.0f} m2")
        print(f"  Precio estimado: {resultado['precio_total']:,.0f} USD")
        print(
            f"  Rango:           {resultado['rango_min']:,.0f} — {resultado['rango_max']:,.0f} USD"
        )
        print(f"  Moneda:          {resultado['moneda']}")
    print("=" * 50)
