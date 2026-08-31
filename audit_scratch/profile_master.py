#!/usr/bin/env python3
"""AeroSpatium master data-layer profiler. Dumps all measured facts to stdout."""
import json, os, struct, sys
import numpy as np

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)

def banner(t):
    p("\n" + "=" * 78); p(t); p("=" * 78)

def fsize(path):
    return os.path.getsize(path)

# =====================================================================
# 0. RAW FILE HEADERS (first bytes / magic numbers)
# =====================================================================
banner("SECTION 0: FILE MAGIC / HEADER BYTES")
magics = {
    "data/raw/dem/srtm_30m_aoi.tif": 16,
    "data/raw/landfire/landfire_evt.tif": 16,
    "data/raw/weather/era5_hourly_aoi.nc": 8,
    "data/raw/osm/aoi.osm.pbf": 32,
}
for path, n in magics.items():
    with open(path, "rb") as f:
        head = f.read(n)
    p(f"{path} [{fsize(path):,} B] first {n} bytes: {head!r}")

# TIFF directory tags of interest via raw parse is handled by rasterio below.

# =====================================================================
# 1. RASTERS
# =====================================================================
import rasterio
from rasterio.transform import array_bounds

for name, path in [("DEM SRTM 30m", "data/raw/dem/srtm_30m_aoi.tif"),
                   ("LANDFIRE EVT", "data/raw/landfire/landfire_evt.tif")]:
    banner(f"RASTER: {name} ({path})")
    with rasterio.open(path) as src:
        p("driver        :", src.driver)
        p("size on disk  :", f"{fsize(path):,}", "bytes")
        p("dtype         :", src.dtypes, "| bands:", src.count)
        p("width x height:", src.width, "x", src.height, "=", src.width * src.height, "px")
        p("crs           :", src.crs.to_string(), "| epsg:", src.crs.to_epsg())
        p("transform     :", tuple(round(v, 10) for v in tuple(src.transform)[:6]))
        p("res (deg)     :", src.res)
        p("bounds        :", tuple(round(v, 6) for v in tuple(src.bounds)))
        p("nodata        :", src.nodata)
        p("block shapes  :", src.block_shapes)
        p("compress      :", src.profile.get("compress", "none"), "| interleave:", src.profile.get("interleave"))
        p("tags          :", dict(src.tags()))
        b = src.read(1)
        mask = src.read_masks(1)
        valid = b[mask == 255]
        p("valid px      :", int(valid.size), "| masked px:", int((mask != 255).sum()))
        p("min/max       :", float(valid.min()), float(valid.max()))
        p("mean/std      :", round(float(valid.mean()), 4), round(float(valid.std()), 4))
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        p("percentiles   :", {q: float(np.percentile(valid, q)) for q in qs})
        u, c = np.unique(valid, return_counts=True)
        p("unique values :", int(u.size))
        if u.size <= 40:
            p("value:count   :", dict(zip(u.tolist(), c.tolist())))
        else:
            top = sorted(zip(u.tolist(), c.tolist()), key=lambda t: -t[1])[:15]
            p("top-15 values :", top)
        # raw nodata sentinel presence in unmasked array
        raw_u = np.unique(b)[:10]
        p("raw min vals  :", raw_u.tolist())
        # corner coords
        for r, cc in [(0, 0), (src.height - 1, src.width - 1)]:
            lon, lat = src.xy(r, cc)
            p(f"corner px({r},{cc}) -> lon={lon:.7f} lat={lat:.7f}")

# =====================================================================
# 2. ERA5-LAND NETCDF
# =====================================================================
import xarray as xr
banner("NETCDF: ERA5-Land (data/raw/weather/era5_hourly_aoi.nc)")
p("size on disk  :", f"{fsize('data/raw/weather/era5_hourly_aoi.nc'):,}", "bytes")
ds = xr.open_dataset("data/raw/weather/era5_hourly_aoi.nc", decode_times=True)
p("engine/format :", ds.encoding.get("source", "?"))
p("dimensions    :", dict(ds.sizes))
p("coords        :", list(ds.coords))
for cname in ds.coords:
    arr = ds[cname]
    p(f"  coord {cname}: dims={arr.dims} dtype={arr.dtype} attrs={dict(arr.attrs)}")
    if arr.size <= 12:
        p(f"    values={arr.values.tolist()}")
    else:
        p(f"    min={arr.values.min()} max={arr.values.max()} first={arr.values.flat[0]} last={arr.values.flat[-1]}")
p("data_vars     :", list(ds.data_vars))
for v in ds.data_vars:
    arr = ds[v]
    p(f"  var {v}: dims={arr.dims} shape={arr.shape} dtype={arr.dtype}")
    p(f"    attrs={dict(arr.attrs)}")
    a = arr.values
    p(f"    min={np.nanmin(a):.4f} max={np.nanmax(a):.4f} mean={np.nanmean(a):.4f} std={np.nanstd(a):.4f}")
tv = ds["time"].values
dts = np.diff(tv).astype("timedelta64[s]").astype(np.int64)
p("time span     :", str(tv.min()), "->", str(tv.max()), "| n =", tv.size)
p("time step(s)  : unique =", np.unique(dts).tolist())
p("global attrs  :", json.dumps({k: str(v) for k, v in ds.attrs.items()}, indent=2)[:2000])
ds.close()

# =====================================================================
# 3. TABULAR CSVs
# =====================================================================
import duckdb
con = duckdb.connect()

def _try_spatial(con):
    try:
        con.execute("INSTALL 'spatial'")
        con.execute("LOAD 'spatial'")
        return True
    except Exception:
        return False

_try_spatial(con)

csvs = {
    "FIRMS VIIRS": ("data/raw/firms/viirs_snpp_california.csv", ","),
    "MODIS": ("data/raw/modis/modis_2023_usa.csv", ","),
    "PeMS traffic": ("data/raw/traffic/pems_station_speeds.csv", ","),
    "AirNow": ("data/raw/airnow/airnow_pm25_aoi.csv", ","),
}
for label, (path, sep) in csvs.items():
    banner(f"CSV: {label} ({path})")
    p("size on disk  :", f"{fsize(path):,}", "bytes")
    with open(path, "rb") as f:
        head = f.read(400)
    p("header bytes  :", head[:300])
    nlines = sum(1 for _ in open(path, "rb"))
    p("physical lines:", nlines, "(incl header)")
    df = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{path}', header=true)").fetchdf()
    p("schema:")
    for _, row in df.iterrows():
        p(f"    {row['column_name']:<22} {row['column_type']:<10} null={row['null']}")
    cnt = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path}')").fetchone()[0]
    p("record count  :", cnt)
    cols = df["column_name"].tolist()
    stats = {}
    for col in cols:
        q = f'SELECT COUNT(*) FILTER (WHERE "{col}" IS NULL) AS nl, MIN("{col}")::VARCHAR mn, MAX("{col}")::VARCHAR mx, COUNT(DISTINCT "{col}") nd FROM read_csv_auto(\'{path}\')'
        try:
            stats[col] = con.execute(q).fetchone()
        except Exception as e:
            stats[col] = ("ERR", str(e)[:60], "", "")
    for col, s in stats.items():
        p(f"    {col:<24} nulls={s[0]:<6} min={str(s[1])[:28]:<28} max={str(s[2])[:28]:<28} distinct={s[3]}")

# FIRMS/MODIS categorical distributions + numeric stats
for label, path, catcols in [
    ("FIRMS VIIRS", "data/raw/firms/viirs_snpp_california.csv", ["satellite", "instrument", "confidence", "daynight"]),
    ("MODIS", "data/raw/modis/modis_2023_usa.csv", ["satellite", "instrument", "confidence", "daynight", "version"]),
]:
    p(f"\n--- {label} categorical distributions ---")
    for cc in catcols:
        try:
            d = con.execute(f'SELECT "{cc}", COUNT(*) n FROM read_csv_auto(\'{path}\') GROUP BY 1 ORDER BY n DESC').fetchall()
            p(f"  {cc}: {d}")
        except Exception as e:
            p(f"  {cc}: ERR {e}")
    for nc in ["latitude", "longitude", "bright_ti4", "frp", "bright_t31", "scan", "track", "acq_time"]:
        try:
            r = con.execute(f'SELECT MIN("{nc}"), MAX("{nc}"), AVG("{nc}"), STDDEV("{nc}") FROM read_csv_auto(\'{path}\')').fetchone()
            p(f"  {nc}: min={r[0]} max={r[1]} mean={round(r[2],4) if r[2] is not None else None} std={round(r[3],4) if r[3] is not None else None}")
        except Exception:
            pass

# duplicate key checks
p("\n--- duplicate detection ---")
p("FIRMS dup (latitude,longitude,acq_date,acq_time):",
  con.execute("SELECT COUNT(*) FROM (SELECT latitude,longitude,acq_date,acq_time,COUNT(*) c FROM read_csv_auto('data/raw/firms/viirs_snpp_california.csv') GROUP BY 1,2,3,4 HAVING c>1)").fetchone()[0])
p("MODIS dup (latitude,longitude,acq_date,acq_time):",
  con.execute("SELECT COUNT(*) FROM (SELECT latitude,longitude,acq_date,acq_time,COUNT(*) c FROM read_csv_auto('data/raw/modis/modis_2023_usa.csv') GROUP BY 1,2,3,4 HAVING c>1)").fetchone()[0])

# PeMS + AirNow full dumps (small files)
p("\n--- PeMS FULL DUMP ---")
p(con.execute("SELECT * FROM read_csv_auto('data/raw/traffic/pems_station_speeds.csv')").fetchdf().to_string())
p("\n--- AirNow FULL DUMP ---")
p(con.execute("SELECT * FROM read_csv_auto('data/raw/airnow/airnow_pm25_aoi.csv')").fetchdf().to_string())

with open("audit_scratch/profile_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nWROTE audit_scratch/profile_out.txt", len(OUT), "lines")
