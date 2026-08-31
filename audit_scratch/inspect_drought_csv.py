#!/usr/bin/env python3
"""Drought shapefile + CSV + GeoJSON profiling."""
import json, numpy as np, duckdb, pyarrow as pa, pyarrow.csv as pacsv

def banner(t): print("\n" + "="*78); print(t); print("="*78)

# ---------------- Drought Shapefile ----------------
banner("SHAPEFILE: Drought USDM  data/raw/drought/USDM_20260818.shp")
con = duckdb.connect()
con.execute("INSTALL spatial"); con.execute("LOAD spatial")
con.execute("CREATE TABLE drought AS SELECT * FROM ST_READ('data/raw/drought/USDM_20260818.shp')")
print("row count   :", con.execute("SELECT COUNT(*) FROM drought").fetchone()[0])
print("dtypes      :", con.execute("DESCRIBE drought").fetchdf().to_string())
rows = con.execute("SELECT * FROM drought LIMIT 5").fetchdf()
print(rows.to_string())
print("column null counts:")
for c in rows.columns:
    print(f"  {c}: ", end="")
print()
allcols = con.execute("SELECT * FROM drought").fetchdf().columns
for c in allcols:
    if c != 'geom':
        nulls = con.execute(f"SELECT COUNT(*) FROM drought WHERE CAST({c} AS VARCHAR)='' OR {c} IS NULL").fetchone()[0]
        print(f"  {c}: nulls/empy={nulls}")
for c in ['DN','RANK','DOMAIN']:
    if c in allcols:
        col = con.execute(f"SELECT {c} FROM drought").fetchdf()[c]
        uniq = col.dropna().unique().tolist()
        print(f"  {c} distribution:", col.value_counts().to_dict())
print("geom type   :", con.execute("SELECT DISTINCT ST_GeometryType(geom) FROM drought").fetchdf().values.tolist())
ex = con.execute("SELECT ST_XMin(geom), ST_YMin(geom), ST_XMax(geom), ST_YMax(geom) FROM drought").fetchone()
print("extent      :", ex)
con.close()

# ---------------- CSV profiles ----------------
banner("CSV profiles via duckdb (memory-mapped, streaming)")
for name,path,sep in [
    ("FIRMS-VIIRS","data/raw/firms/viirs_snpp_california.csv",","),
    ("MODIS","data/raw/modis/modis_2023_usa.csv",","),
    ("Traffic-PeMS","data/raw/traffic/pems_station_speeds.csv",","),
    ("AirNow","data/raw/airnow/airnow_pm25_aoi.csv",","),
]:
    con = duckdb.connect()
    con.execute("INSTALL spatial"); con.execute("LOAD spatial")
    rel = con.execute(f"SELECT * FROM read_csv_auto('{path}')")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{path}')").fetchdf()
    print(f"\n>>> {name}: {path}")
    print("schema:\n", schema.to_string())
    n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path}')").fetchone()[0]
    print("row count:", n)
    # preview
    df = con.execute(f"SELECT * FROM read_csv_auto('{path}') LIMIT 3").fetchdf()
    print(df.to_string())
    # null counts
    cols = list(df.columns)
    nl = con.execute(f"SELECT {','.join([f'COUNT({c}) FILTER ({c} IS NULL) AS {c}' for c in cols])} FROM read_csv_auto('{path}')").fetchone()
    print("null counts:", dict(zip(cols, list(nl))))
    con.close()
