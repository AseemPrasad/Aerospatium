#!/usr/bin/env python3
"""Smoke-test every ingestion snippet destined for the audit report."""
import time
import numpy as np

t0 = time.time()

# --- SNIPPET 1: FIRMS/MODIS typed streaming read w/ bbox pushdown ---
import duckdb
con = duckdb.connect()
con.execute("INSTALL 'spatial'"); con.execute("LOAD 'spatial'")
modis_aoi = con.execute("""
    SELECT latitude, longitude, acq_date, acq_time::INT AS acq_hhmm,
           satellite, confidence::TINYINT AS conf, frp::FLOAT4 AS frp_mw, daynight, type::TINYINT AS fire_type
    FROM read_csv('data/raw/modis/modis_2023_usa.csv',
                  types={'acq_time':'VARCHAR','confidence':'INTEGER','type':'INTEGER'},
                  timestampformat='%Y-%m-%d')
    WHERE longitude BETWEEN -121.8 AND -121.3 AND latitude BETWEEN 39.6 AND 40.0
""").fetchall()
print("S1 MODIS AOI rows:", len(modis_aoi), "| sample:", modis_aoi[0])

# --- SNIPPET 2: rasterio windowed read (RAM-bounded) ---
import rasterio
from rasterio.windows import from_bounds
with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as src:
    win = from_bounds(-121.75, 39.88, -121.70, 39.92, src.transform)
    dem_win = src.read(1, window=win)
    print("S2 DEM window:", dem_win.shape, dem_win.dtype, f"mean={dem_win.mean():.1f}m")

# --- SNIPPET 3: xarray disk-slice ---
import xarray as xr
ds = xr.open_dataset("data/raw/weather/era5_hourly_aoi.nc")
sl = ds[["u10", "v10", "t2m", "d2m"]].sel(time=slice("2023-08-15T06", "2023-08-15T12"),
                                          latitude=slice(39.9, 39.6), longitude=slice(-121.8, -121.5))
ws10 = np.sqrt(sl.u10**2 + sl.v10**2)
rh = np.exp(17.625 * (sl.d2m - 273.15) / (243.04 + (sl.d2m - 273.15))) / \
     np.exp(17.625 * (sl.t2m - 273.15) / (234.18 + (sl.t2m - 273.15))) * 100
print(f"S3 ERA5 slice dims={dict(sl.sizes)} | ws10 mean={float(ws10.mean()):.2f} m/s | RH mean={float(rh.mean()):.1f}%")
ds.close()

# --- SNIPPET 4: USDM MakeValid + AOI intersect ---
usdm = con.execute("""
    SELECT DM::INT AS dm, ST_MakeValid(geom) AS geom
    FROM ST_Read('data/raw/drought/USDM_20260818.shp')
""").df()
hit = con.execute("""
    WITH v AS (SELECT DM::INT dm, ST_MakeValid(geom) geom FROM ST_Read('data/raw/drought/USDM_20260818.shp'))
    SELECT dm FROM v WHERE ST_Intersects(geom, ST_GeomFromText('POLYGON((-121.8 39.6,-121.3 39.6,-121.3 40.0,-121.8 40.0,-121.8 39.6))'))
    ORDER BY dm DESC LIMIT 1
""").fetchone()
print("S4 USDM valid rows:", len(usdm), "| max severity touching AOI: DM =", hit[0])

# --- SNIPPET 5: GeoJSON vector reads ---
import geopandas as gpd
cen = gpd.read_file("data/raw/census/census_blocks.geojson")
wf = gpd.read_file("data/raw/wfigs/wfigs_perimeters.geojson")
print("S5 census:", cen.crs, cen.shape, "| wfigs:", wf.crs, wf.shape)

# --- SNIPPET 6: H3 assignment of MODIS points ---
import h3
hids = [h3.latlng_to_cell(lat, lon, 8) for lat, lon, *_ in modis_aoi]
print("S6 h3 res8 ids:", len(set(hids)), "unique hexes for", len(hids), "MODIS pts")

# --- SNIPPET 7: OSM filtered parse (documents the broken extract) ---
import osmium as o
class RdHwy(o.SimpleHandler):
    def __init__(self):
        super().__init__(); self.n = 0
    def way(self, w):
        if any(t.k == "highway" for t in w.tags):
            self.n += 1
h = RdHwy(); h.apply_file("data/raw/osm/aoi.osm.pbf", locations=True)
print("S7 OSM highway ways parsed:", h.n, "(NOTE: extract covers Sacramento, NOT AOI)")

print(f"ALL SNIPPETS OK in {time.time()-t0:.1f}s")
