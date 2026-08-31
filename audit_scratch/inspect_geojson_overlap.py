#!/usr/bin/env python3
"""GeoJSON + coordinate overlap + drought full dump."""
import json, numpy as np, duckdb
from shapely.geometry import shape

def banner(t): print("\n" + "="*78); print(t); print("="*78)

# ---------- Drought full dump ----------
banner("DROUGHT USDM full dump (5 features)")
con = duckdb.connect()
con.execute("INSTALL spatial"); con.execute("LOAD spatial")
con.execute("CREATE TABLE drought AS SELECT * FROM ST_READ('data/raw/drought/USDM_20260818.shp')")
df = con.execute("SELECT OGC_FID,OBJECTID,DM,Shape_Leng,Shape_Area FROM drought").fetchdf()
print(df.to_string())
print("DM categorical distribution:")
print(df['DM'].value_counts().sort_index().to_dict())
print("Shape_Leng range:", df['Shape_Leng'].min(), df['Shape_Leng'].max())
print("Shape_Area range:", df['Shape_Area'].min(), df['Shape_Area'].max())
con.close()

# ---------- GeoJSON helper ----------
def geojson_profile(name, path):
    banner(f"GEOJSON: {name}  {path}")
    with open(path) as f:
        gj = json.load(f)
    feats = gj['features']
    print("type:", gj.get('type'))
    print("crs:", gj.get('crs'))
    print("feature count:", len(feats))
    if feats:
        props = feats[0]['properties']
        print("property keys:", list(props.keys()))
        geom_types = set(f['geometry']['type'] for f in feats)
        print("geometry types:", geom_types)
        # property schema
        print("properties schema:")
        for k,v in props.items():
            sample = [f['properties'].get(k) for f in feats]
            print(f"   {k}: {v!r}  values_sample={sample}")
        # geometry extent
        lons, lats = [], []
        for f in feats:
            g = shape(f['geometry'])
            # get bounds
            b = g.bounds
            lons += [b[0], b[2]]; lats += [b[1], b[3]]
        print(f"extent: lon({min(lons):.5f},{max(lons):.5f}) lat({min(lats):.5f},{max(lats):.5f})")
        for f in feats[:3]:
            g = shape(f['geometry'])
            print("  feat props:", f['properties'], "| geom type:", f['geometry']['type'], "| bounds:", g.bounds)

geojson_profile("WFIGS perimeters", "data/raw/wfigs/wfigs_perimeters.geojson")
geojson_profile("Census blocks", "data/raw/census/census_blocks.geojson")

# ---------- Coordinate overlap checks ----------
banner("COORDINATE OVERLAP vs AOI")
con = duckdb.connect()
con.execute("INSTALL spatial"); con.execute("LOAD spatial")
AOI = (-121.8, 39.6, -121.55, 40.0)  # approx matching raster/netcdf
# FIRMS
for name,path,lat,lon in [
  ("FIRMS-VIIRS","data/raw/firms/viirs_snpp_california.csv","latitude","longitude"),
  ("MODIS","data/raw/modis/modis_2023_usa.csv","latitude","longitude"),
  ("AirNow","data/raw/airnow/airnow_pm25_aoi.csv","latitude","longitude"),
]:
    r = con.execute(f"SELECT MIN({lat}),MAX({lat}),MIN({lon}),MAX({lon}) FROM read_csv_auto('{path}')").fetchone()
    print(f"{name}: lat[{r[0]},{r[1]}] lon[{r[2]},{r[3]}]")
# Traffic stations (lat/lon? check cols - actually station_id only, no lat/lon). Inspect.
print("Traffic columns:", con.execute("DESCRIBE SELECT * FROM read_csv_auto('data/raw/traffic/pems_station_speeds.csv')").fetchdf()['column_name'].tolist())
# WFIGS / Census extents via geojson (already printed above)
con.close()
