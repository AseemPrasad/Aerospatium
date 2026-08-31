#!/usr/bin/env python3
"""v2 re-audit: full-AOI rasters + fuel crosswalk validation."""
import os
import numpy as np
import rasterio
from rasterio.transform import array_bounds

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s); print(s)
def banner(t):
    p("\n" + "=" * 78); p(t); p("=" * 78)

AOI = (-121.8, 39.6, -121.3, 40.0)

for name, path in [("DEM SRTM 30m", "data/raw/dem/srtm_30m_aoi.tif"),
                   ("LANDFIRE EVT", "data/raw/landfire/landfire_evt.tif")]:
    banner(f"RASTER v2: {name} ({path})")
    with rasterio.open(path) as src:
        p("size on disk  :", f"{os.path.getsize(path):,}", "bytes")
        p("driver        :", src.driver, "| dtype:", src.dtypes, "| bands:", src.count)
        p("width x height:", src.width, "x", src.height, "=", f"{src.width*src.height:,}", "px")
        p("crs           :", src.crs.to_string(), "| epsg:", src.crs.to_epsg())
        p("transform     :", tuple(round(v, 10) for v in tuple(src.transform)[:6]))
        p("res (deg)     :", src.res)
        b = tuple(round(v, 7) for v in tuple(src.bounds))
        p("bounds        :", b)
        p("nodata        :", src.nodata)
        p("block shapes  :", src.block_shapes)
        p("compress      :", src.profile.get("compress", "none"))
        p("tags          :", dict(src.tags()))
        # coverage proof vs canonical AOI
        rbox = array_bounds(src.height, src.width, src.transform)
        cov_ok = (abs(rbox[0]-AOI[0]) < 1e-9 and abs(rbox[1]-AOI[1]) < 1e-9
                  and abs(rbox[2]-AOI[2]) < 1e-9 and abs(rbox[3]-AOI[3]) < 1e-9)
        p("EXACT AOI match (tol 1e-9):", cov_ok)
        arr = src.read(1)
        m = src.read_masks(1)
        valid = arr[m == 255]
        p("valid px      :", int(valid.size), "| masked px:", int((m != 255).sum()))
        p("min/max       :", float(valid.min()), float(valid.max()))
        p("mean/std      :", round(float(valid.mean()), 4), round(float(valid.std()), 4))
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
        p("percentiles   :", {q: round(float(np.percentile(valid, q)), 3) for q in qs})
        u, c = np.unique(valid, return_counts=True)
        p("unique values :", int(u.size))
        if u.size <= 20:
            p("value:count   :", dict(zip(u.tolist(), c.tolist())))
            p("share %       :", {int(k): round(100*v/valid.size, 2) for k, v in zip(u.tolist(), c.tolist())})
        else:
            top = sorted(zip(u.tolist(), c.tolist()), key=lambda t: -t[1])[:8]
            p("top-8 values  :", [(round(k, 2), v) for k, v in top])
        # nodata sentinel presence in raw values
        p("raw contains -9999:", bool((arr == -9999).any()), "| raw contains -32768:", bool((arr == -32768).any()))
        # corner coords
        for r, cc in [(0, 0), (src.height-1, src.width-1)]:
            lon, lat = src.xy(r, cc)
            p(f"corner px({r},{cc}) -> lon={lon:.7f} lat={lat:.7f}")
        # co-registration check vs the other raster handled below

banner("CO-REGISTRATION CHECK (DEM vs EVT)")
with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as a, rasterio.open("data/raw/landfire/landfire_evt.tif") as e:
    same = (a.transform == e.transform and a.crs == e.crs and (a.width, a.height) == (e.width, e.height))
    p("identical transform/crs/grid:", same)

banner("FUEL CROSSWALK CSV")
cw_path = "data/raw/landfire/fuel_crosswalk.csv"
p("size on disk  :", f"{os.path.getsize(cw_path):,}", "bytes")
with open(cw_path, "rb") as f:
    p("raw bytes     :", f.read())
import duckdb
con = duckdb.connect()
df = con.execute(f"SELECT * FROM read_csv_auto('{cw_path}', header=true)").df()
p("shape         :", df.shape)
p("dtypes        :", {k: str(v) for k, v in df.dtypes.items()})
p(df.to_string())
p("null counts   :", df.isna().sum().to_dict())
codes_raster = None
with rasterio.open("data/raw/landfire/landfire_evt.tif") as src:
    codes_raster = set(int(x) for x in np.unique(src.read(1)))
codes_cw = set(int(x) for x in df.iloc[:, 0])
p("raster codes  :", sorted(codes_raster))
p("crosswalk keys:", sorted(codes_cw))
p("MISSING in crosswalk:", sorted(codes_raster - codes_cw), "| EXTRA in crosswalk:", sorted(codes_cw - codes_raster))

# H3 res-8 coverage estimate over full AOI
import h3, math
res = 8
cell_km2 = h3.cell_area(h3.latlng_to_cell(39.8, -121.55, res), unit='km^2')
aoi_km2 = 0.2 * 111.32**2 * math.cos(math.radians(39.8))
p("\nH3 res-8: hex area %.4f km2 | AOI %.1f km2 | est cells %d" % (cell_km2, aoi_km2, round(aoi_km2/cell_km2)))
# px per hex now
px_km2 = 0.000277**2 * 111.32**2 * math.cos(math.radians(39.8))
p("px area ~%.0f m2 -> ~%.0f px/hex" % (px_km2*1e6, cell_km2*1e6/(px_km2*1e6)))

with open("audit_scratch/v2_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nWROTE audit_scratch/v2_out.txt")
