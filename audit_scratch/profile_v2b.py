#!/usr/bin/env python3
"""v2 gap quantification + regression spot-checks + NB01 pipeline validation."""
import numpy as np, math, json
import rasterio
import h3

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s); print(s)

AOI_W, AOI_S, AOI_E, AOI_N = -121.8, 39.6, -121.3, 40.0

# ---- 1. Quantify raster/AOI sliver ----
with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as src:
    b = src.bounds
    p("raster bounds :", tuple(round(v,7) for v in tuple(b)))
    gap_e_deg = AOI_E - b.right          # positive => shortfall
    gap_s_deg = b.bottom - AOI_S
    m_per_deg_lon = 111320*math.cos(math.radians(39.8))
    p(f"east gap : {gap_e_deg:.7f} deg = {gap_e_deg*m_per_deg_lon:.1f} m")
    p(f"south gap: {gap_s_deg:.7f} deg = {gap_s_deg*110574:.1f} m")
    cov = (b.right-b.left)*(b.top-b.bottom) / ((AOI_E-AOI_W)*(AOI_N-AOI_S))
    p(f"strict area coverage of AOI: {cov*100:.2f}%")
    # px needed for exact fit
    import math as _m
    p("px needed exact E-W:", round((AOI_E-AOI_W)/0.000277, 2), "| provided:", src.width)
    p("px needed exact N-S:", round((AOI_N-AOI_S)/0.000277, 2), "| provided:", src.height)

# ---- 2. H3 res-8 cells whose CENTROID falls in the sliver ----
def aoi_hex_centroids(res=8):
    ring = [(AOI_S, AOI_W), (AOI_S, AOI_E), (AOI_N, AOI_E), (AOI_N, AOI_W)]  # (lat, lng)
    return h3.polygon_to_cells(h3.LatLngPoly(ring), res)

cells = aoi_hex_centroids(8)
p("\nH3 res-8 cells intersecting AOI bbox:", len(cells))
affected = []
for c in cells:
    lat, lon = h3.cell_to_latlng(c)
    if lon > (-121.3014) or lat < 39.60112:
        affected.append((c, round(lat,6), round(lon,6)))
p("cells with centroid OUTSIDE raster coverage:", len(affected))
for c, la, lo in affected[:10]:
    p("   ", c, la, lo)

# ---- 3. Regression spot-checks (F-1, F-3 unchanged?) ----
import osmium as o
rd = o.io.Reader('data/raw/osm/aoi.osm.pbf')
hb = rd.header().box(); rd.close()
p("\nOSM header box (v2 recheck):", hb.bottom_left, "->", hb.top_right)

import duckdb
con = duckdb.connect()
n_firms = con.execute("""SELECT COUNT(*) FILTER (WHERE longitude BETWEEN -121.8 AND -121.3 AND latitude BETWEEN 39.6 AND 40.0),
                                COUNT(*) FROM read_csv_auto('data/raw/firms/viirs_snpp_california.csv')""").fetchone()
p(f"FIRMS AOI hits (v2 recheck): {n_firms[0]} / {n_firms[1]}")
n_modis = con.execute("""SELECT COUNT(*) FILTER (WHERE longitude BETWEEN -121.8 AND -121.3 AND latitude BETWEEN 39.6 AND 40.0),
                                COUNT(*) FROM read_csv_auto('data/raw/modis/modis_2023_usa.csv')""").fetchone()
p(f"MODIS AOI hits (v2 recheck): {n_modis[0]} / {n_modis[1]}")

# ---- 4. NB01 validated pipeline: full-AOI read + crosswalk join + per-hex aggregate ----
p("\n--- NB01 PIPELINE TEST ---")
with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as dem_src, \
     rasterio.open("data/raw/landfire/landfire_evt.tif") as evt_src:
    assert dem_src.transform == evt_src.transform and dem_src.shape == evt_src.shape
    dem = dem_src.read(1); evt = evt_src.read(1)
    p("stacked arrays:", dem.shape, dem.dtype, evt.dtype, f"| RAM ~{dem.nbytes/1e6+evt.nbytes/1e6:.1f} MB")

cw = con.execute("SELECT * FROM read_csv_auto('data/raw/landfire/fuel_crosswalk.csv')").df()
lut = cw.set_index("evt_code")
burnable = lut.loc[lut.index != 11]
p("burnable codes:", burnable.index.tolist(), "| non-burnable:", [11])

# per-hex mode of EVT + mean elevation via sampled centroids (vectorized demo on subset)
sample_n = 5000
rng = np.random.default_rng(42)
rr = rng.integers(0, dem.shape[0], sample_n); cc = rng.integers(0, dem.shape[1], sample_n)
lons = -121.8 + (cc + 0.5)*0.000277; lats = 40.0 - (rr + 0.5)*0.000277
hexes = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(lats, lons)]
import collections
hex_mode = collections.defaultdict(collections.Counter)
for hx, code in zip(hexes, evt[rr, cc]):
    hex_mode[hx][int(code)] += 1
demo_hex = next(iter(hex_mode))
maj_code = hex_mode[demo_hex].most_common(1)[0]
params = lut.loc[maj_code[0]]
p(f"demo hex {demo_hex}: majority EVT={maj_code[0]} ({maj_code[1]} px) -> "
  f"{params.fuel_name} FBFM{params.fbfm13} load={params.fuel_load_kg_m2} kg/m2 sav={params.sav_ratio_m_inv} 1/m depth={params.fuel_bed_depth_m} m Mx={params.moisture_extinction}")
p("PIPELINE OK")

with open("audit_scratch/v2_checks.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nWROTE audit_scratch/v2_checks.txt")
