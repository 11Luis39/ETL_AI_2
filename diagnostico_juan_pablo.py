# Pegalo en un script temporal y correlo
import geopandas as gpd

edges = gpd.read_file("data/geo/red_vial_scz.gpkg")

if edges.crs.to_epsg() != 4326:
    edges = edges.to_crs("EPSG:4326")

edges["nombres_lista"] = edges["name"].apply(
    lambda val: [str(v).strip().lower() for v in val] if isinstance(val, list)
    else ([p.strip().lower() for p in val.replace(";",",").split(",")] if isinstance(val, str) else [])
)

mask = edges["nombres_lista"].apply(lambda l: "avenida blooming" in l)
segs = edges[mask].copy()

print(f"Segmentos encontrados: {len(segs)}")
print("\nCoordenadas de cada segmento:")
for i, row in segs.iterrows():
    bounds = row.geometry.bounds  # (minx, miny, maxx, maxy) = (lng_min, lat_min, lng_max, lat_max)
    print(f"  lng: {bounds[0]:.4f} → {bounds[2]:.4f}  |  lat: {bounds[1]:.4f} → {bounds[3]:.4f}")