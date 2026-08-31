#!/usr/bin/env python3
"""AOI overlap proof + H3 res-8 discretization checks."""
import json, math
import numpy as np
import duckdb
import rasterio
import xarray as xr
from shapely.geometry import shape, Point, box
import h3

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s); print(s)
def banner(t):
    p("\n" + "=" * 78); p(t); p("=" * 78)

AOI = box(-121.8, 39.6, -121.3, 40.0)   # canonical AOI from census tiling
p("CANONICAL AOI:", AOI.bounds, "| area deg2:", round(AOI.area, 4),
  "| approx km2:", round(AOI.area * 111.32**2 * math.cos(math.radians(39.8)), 1))

# ---- OSM header box + precise way-in-AOI test ----
import osmium as o
rd = o.io.Reader('data/raw/osm/aoi.osm.pbf')
hb = rd.header().box()
rd.close()
try:
    p("OSM PBF header box:", hb.bottom_left, "->", hb.top_right)
except Exception as e:
    p("header box err:", e)

class W(o.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.total_ways = 0; self.in_aoi = 0; self.sample_in = []
    def way(self, w):
        self.total_ways += 1
        try:
            locs = [nd.location for nd in w.nodes]
        except Exception:
            return
        for l in locs[:50]:
            if l.valid() and 39.6 <= l.lat <= 40.0 and -121.8 <= l.lon <= -121.3:
                self.in_aoi += 1
                if len(self.sample_in) < 3:
                    self.sample_in.append({t.k: t.v for t in w.tags})
                break
w = W(); w.apply_file('data/raw/osm/aoi.osm.pbf', locations=True)
p(f"OSM total ways={w.total_ways:,} | ways intersecting AOI={w.in_aoi} ({100*w.in_aoi/max(w.total_ways,1):.4f}%)")
p("sample in-AOI ways:", w.sample_in[:1])

# ---- Raster overlap ----
for name, path in [("DEM", "data/raw/dem/srtm_30m_aoi.tif"), ("EVT", "data/raw/landfire/landfire_evt.tif")]:
    with rasterio.open(path) as src:
        b = src.bounds
        rbox = box(b.left, b.bottom, b.right, b.top)
        inter = rbox.intersection(AOI)
        cov = inter.area / AOI.area * 100
        p(f"{name}: bounds={tuple(round(v,4) for v in tuple(b))} -> AOI coverage {cov:.2f}% | overlap bbox={tuple(round(v,4) for v in inter.bounds)}")

# ---- ERA5 grid coverage ----
ds = xr.open_dataset("data/raw/weather/era5_hourly_aoi.nc")
lo = ds.longitude.values; la = ds.latitude.values
ebox = box(lo.min(), la.min(), lo.max(), la.max())
p(f"ERA5 cell-center extent: lon({lo.min()},{lo.max()}) lat({la.min()},{la.max()}) step_lon={np.diff(lo)[0]:.5f} step_lat={np.diff(la)[0]:.5f}")
p(f"ERA5 half-cell-padded extent: lon({lo.min()-0.02778:.5f},{lo.max()+0.02778:.5f}) lat({la.min()-0.02222:.5f},{la.max()+0.02222:.5f})")
p("ERA5 covers AOI:", ebox.contains(AOI), "| padded covers AOI:",
  box(lo.min()-0.02778, la.min()-0.02222, lo.max()+0.02778, la.max()+0.02222).contains(AOI))
ds.close()

# ---- Point feeds inside AOI ----
con = duckdb.connect()
con.execute("INSTALL 'spatial'"); con.execute("LOAD 'spatial'")
for label, path, latc, lonc in [
    ("FIRMS VIIRS", "data/raw/firms/viirs_snpp_california.csv", "latitude", "longitude"),
    ("MODIS", "data/raw/modis/modis_2023_usa.csv", "latitude", "longitude"),
]:
    r = con.execute(f"""SELECT COUNT(*) tot,
        COUNT(*) FILTER (WHERE {lonc} BETWEEN -121.8 AND -121.3 AND {latc} BETWEEN 39.6 AND 40.0) in_aoi,
        COUNT(*) FILTER (WHERE {lonc} BETWEEN -122.5 AND -120.5 AND {latc} BETWEEN 39.0 AND 40.5) in_butte_buffer
        FROM read_csv_auto('{path}')""").fetchone()
    p(f"{label}: total={r[0]:,} | strictly inside AOI={r[1]} | inside Butte buffer(Â±1Â°)={r[2]}")

# AirNow stations vs AOI
p("\nAirNow station containment:")
rows = con.execute("SELECT site_name, latitude, longitude FROM read_csv_auto('data/raw/airnow/airnow_pm25_aoi.csv')").fetchall()
for name_, lat_, lon_ in rows:
    inside = AOI.contains(Point(lon_, lat_))
    dist_km = AOI.distance(Point(lon_, lat_)) * 111.32
    p(f"   {name_:<20} ({lat_},{lon_}) inside_AOI={inside} dist_outside_deg={round(AOI.distance(Point(lon_,lat_)),4)}")

# Census/WFIGS containment
gj = json.load(open("data/raw/census/census_blocks.geojson"))
for f in gj["features"]:
    g = shape(f["geometry"])
    p(f"Census BG {f['properties']['GEOID']}: equals_AOI_quadrant={g.equals_exact(g.intersection(AOI),1e-9)} within_AOI={AOI.contains(g)}")
wf = shape(json.load(open("data/raw/wfigs/wfigs_perimeters.geojson"))["features"][0]["geometry"])
p(f"WFIGS perimeter within_AOI={AOI.contains(wf)} | frac_of_AOI={wf.area/AOI.area*100:.2f}%")

# Drought: which DM polygons contain AOI centroid?
cx, cy = (-121.55, 39.8)
rows = con.execute("""SELECT DM::INT dm, ST_Contains(geom, ST_Point(-121.55, 39.8)) c FROM 'data/raw/drought/USDM_20260818.shp' ORDER BY dm""").fetchall()
p("USDM contains AOI centroid(-121.55,39.8):", rows)
# valid-area calc in equal-area
rows = con.execute("""SELECT DM::INT dm, ROUND(ST_Area(ST_Transform(geom,'EPSG:4326','EPSG:6933'))/1e12, 4) area_mkm2
                      FROM 'data/raw/drought/USDM_20260818.shp' ORDER BY dm""").fetchall()
p("USDM areas (million kmÂ², EPSG:6933):", rows)
# make-valid test on DM=2
r = con.execute("""SELECT ST_IsValid(ST_MakeValid(geom)), ST_NumGeometries(ST_MakeValid(geom)),
                   ST_NumGeometries(geom) FROM 'data/raw/drought/USDM_20260818.shp' WHERE DM=2""").fetchone()
p("DM=2 after MakeValid: valid=", r[0], "ngeom_after=", r[1], "ngeom_before=", r[2])

# =====================================================================
# H3 RESOLUTION 8 DISCRETIZATION MATRIX
# =====================================================================
banner("H3 RES-8 MAPPING CHECKS")
res = 8
cell = h3.cell_area(h3.latlng_to_cell(39.8, -121.55, res), unit='km^2')
edge = h3.average_hexagon_edge_length(res, unit='km')
p(f"H3 res-8: avg edge={edge:.4f} km | avg hex area={cell:.4f} kmÂ² | cells needed over AOI â‰ˆ {int(round(AOI.area*111.32**2*math.cos(math.radians(39.8))/cell)):,}")
demo = {
    "AOI centroid": (-121.55, 39.8),
    "Paradise (AirNow)": (39.759, -121.611),
    "WFIGS perimeter centroid": (wf.centroid.x, wf.centroid.y),
}
for k, (lon_, lat_) in demo.items():
    p(f"  {k:<26} -> h3_id={h3.latlng_to_cell(lat_, lon_, res)} int={h3.str_to_int(h3.latlng_to_cell(lat_, lon_, res))}")

# raster center -> h3 sample
with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as src:
    lon_c = (src.bounds.left + src.bounds.right)/2; lat_c = (src.bounds.top + src.bounds.bottom)/2
    p(f"  DEM/EVT grid center      -> h3_id={h3.latlng_to_cell(lat_c, lon_c, res)}")
    # how many DEM pixels per hex (~0.74 kmÂ²): px area
    px_km2 = abs(src.transform.a) * abs(src.transform.e) * (111.32**2) * math.cos(math.radians(lat_c))
    p(f"  DEM pixel area â‰ˆ {px_km2*1e6:.0f} mÂ² -> ~{cell*1e6/(px_km2*1e6):,.0f} px per hex-cell")

with open("audit_scratch/align_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nWROTE audit_scratch/align_out.txt")

