#!/usr/bin/env python3
"""Binary/raster inspection for AeroSpatium audit."""
import json, sys, os
import numpy as np
import rasterio
from rasterio.transform import array_bounds
import xarray as xr
import duckdb

def banner(t): print("\n" + "="*78); print(t); print("="*78)

# ---------------- DEM ----------------
banner("RASTER 1: DEM  data/raw/dem/srtm_30m_aoi.tif")
with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as src:
    print("driver       :", src.driver)
    print("dtype        :", src.dtypes)
    print("count(bands) :", src.count)
    print("width/height :", src.width, src.height)
    print("crs          :", src.crs)
    print("transform    :", list(src.transform))
    print("bounds       :", src.bounds)
    print("res          :", src.res)
    nodata = src.nodata
    print("nodata       :", nodata)
    b = src.read(1, masked=True)
    print("shape        :", b.shape)
    print("min/max      :", float(np.nanmin(b)), float(np.nanmax(b)))
    print("mean/std     :", float(np.nanmean(b)), float(np.nanstd(b)))
    # value distribution percentiles
    flat = b.compressed()
    if flat.size:
        for p in [0,1,5,25,50,75,95,99,100]:
            print(f"  p{p:<3}     :", float(np.percentile(flat, p)))
    print("unique count :", len(np.unique(flat)) if flat.size else 0)
    print("n valid px   :", int(b.count()))
    print("n nodata px  :", int((b.mask).sum()) if b.mask is not False else 0)
    # corner coordinates
    ys, xs = src.height, src.width
    corners = [(0,0),(0,xs-1),(ys-1,0),(ys-1,xs-1)]
    print("corner lon/lat:")
    for r,c in corners:
        lon,lat = src.xy(r,c)
        print(f"   px({r},{c}) -> lon={lon:.6f} lat={lat:.6f}")
    tags = src.tags()
    print("tags         :", dict(tags))

# ---------------- LANDFIRE EVT ----------------
banner("RASTER 2: LANDFIRE EVT  data/raw/landfire/landfire_evt.tif")
with rasterio.open("data/raw/landfire/landfire_evt.tif") as src:
    print("driver       :", src.driver)
    print("dtype        :", src.dtypes)
    print("count(bands) :", src.count)
    print("width/height :", src.width, src.height)
    print("crs          :", src.crs)
    print("transform    :", list(src.transform))
    print("bounds       :", src.bounds)
    print("res          :", src.res)
    nodata = src.nodata
    print("nodata       :", nodata)
    b = src.read(1, masked=True)
    flat = b.compressed()
    print("min/max      :", float(np.nanmin(b)), float(np.nanmax(b)))
    print("mean/std     :", float(np.nanmean(b)), float(np.nanstd(b)))
    u, cnt = np.unique(flat, return_counts=True)
    print("unique categorical codes (value: count):")
    for v,c in zip(u.tolist(), cnt.tolist()):
        print(f"   {v:<6} : {c}")
    print("n valid px   :", int(b.count()))
    print("n nodata px  :", int((b.mask).sum()) if b.mask is not False else 0)
    print("tags         :", dict(src.tags()))

# ---------------- NetCDF Weather ----------------
banner("NETCDF: Weather  data/raw/weather/era5_hourly_aoi.nc")
ds = xr.open_dataset("data/raw/weather/era5_hourly_aoi.nc")
print("dimensions:", dict(ds.dims))
print("coords    :", list(ds.coords))
for c in ds.coords:
    arr = ds[c]
    if arr.size <= 10:
        print(f"  coord {c}: dims={arr.dims} vals={arr.values.tolist()}")
print("variables :", list(ds.data_vars))
for v in ds.data_vars:
    arr = ds[v]
    print(f"  var {v}: dims={arr.dims} shape={arr.shape} dtype={arr.dtype} units={arr.attrs.get('units','NA')} long_name={arr.attrs.get('long_name','NA')}")
    if arr.size <= 8:
        print(f"     vals={arr.values.tolist()}")
print("global attrs:")
for k,v in ds.attrs.items():
    print(f"  {k}: {v}")
print("time span :", str(ds["time"].values.min()), "->", str(ds["time"].values.max()))
dt = np.diff(ds["time"].values).astype("timedelta64")
print("time freq :", str(dt[0]) if len(dt) else "n/a", "median:", str(np.median(dt)))
# value stats for each var
for v in ds.data_vars:
    a = ds[v].values
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.integer):
        print(f"  {v} stats: min={np.nanmin(a)} max={np.nanmax(a)} mean={np.nanmean(a):.4f} std={np.nanstd(a):.4f}")
ds.close()

# ---------------- Drought Shapefile ----------------
banner("SHAPEFILE: Drought USDM  data/raw/drought/USDM_20260818.shp")
con = duckdb.connect()
con.execute("INSTALL('spatial')")
con.execute("LOAD('spatial')")
try:
    con.execute("CREATE TABLE drought AS SELECT * FROM ST_READ('data/raw/drought/USDM_20260818.shp')")
    print("row count   :", con.execute("SELECT COUNT(*) FROM drought").fetchone()[0])
    print("columns     :", con.execute("DESCRIBE drought").fetchdf().to_dict('records'))
    rows = con.execute("SELECT * FROM drought LIMIT 3").fetchdf()
    print(rows.to_string())
    print("categorical distribution of severity (dn field):")
    df = con.execute("SELECT DN, COUNT(*) c FROM drought GROUP BY DN ORDER BY DN").fetchdf()
    print(df.to_string())
    print("geometry extent (WGS84):")
    ex = con.execute("SELECT ST_XMin(geom) xmin, ST_YMin(geom) ymin, ST_XMax(geom) xmax, ST_YMax(geom) ymax FROM drought LIMIT 1").fetchone()
    print(ex)
    print("geom type   :", con.execute("SELECT DISTINCT ST_GeometryType(geom) FROM drought").fetchdf().to_string())
except Exception as e:
    print("ERR", e)
    # fallback: read via the .prj-known crs; try fiona-like via duckdb
    import rasterio
    print("Attempting direct read without spatial...")
con.close()
