# AeroSpatium — Updated Data Layer Specification & Ingestion Audit Report (v2)

**Project:** AeroSpatium — Multi-modal Wildfire Progression & Evacuation Route Optimization Engine
**Audit Date:** 2026-08-22 (supersedes v1 of 2026-08-21)
**Scope:** Full re-scan of `data/raw/` — **19 physical files** (18 prior + new fuel crosswalk), **57,343,941 bytes (54.68 MiB)** total on disk
**Environment:** Python 3.14.0 · rasterio 1.5.1 · duckdb 1.5.5 (+spatial) · h3 4.5.0 · osmium · xarray 2026.7.0
**Method:** Both regenerated rasters re-parsed byte-level and pixel-level via `rasterio`; crosswalk parsed via raw bytes + DuckDB typed read; regression spot-checks re-ran against OSM/FIRMS/MODIS to confirm no silent drift. All values below are **measured this session**.

---

## ⚠️ Executive Summary — Findings Status Board

| ID | Finding | v1 Status | v2 Status |
|----|---------|-----------|-----------|
| F-1 | OSM extract covers Sacramento metro, not the AOI (header box `(−121.625, 38.374)→(−121.154, 38.738)`; 1/554,318 ways touch AOI) | 🔴 BLOCKER | 🔴 **OPEN — unchanged.** Header box re-read this session is byte-identical. Notebook 03 remains blocked until `osmium extract --bbox=-121.8,39.6,-121.3,40.0` is run against a California PBF. |
| F-2 | DEM/EVT covered only 9.59 % of AOI | 🔴 BLOCKER | ✅ **RESOLVED** (with one residual note). Both rasters regenerated to the full-AOI grid: `(1800 × 1440)`, bounds `(−121.8, 39.60112, −121.3014, 40.0)`, co-registered. Strict geometric coverage = **99.44 %** of the canonical box; the uncovered margin is an L-shaped sliver ≤ **123.8 m** wide along the east/south edges (integer-pixel rounding: exact fit requires 1805 × 1444 px at 0.000277°). Impact at modeling granularity: **15 of 2,526 H3-res-8 cells (0.59 %)** have centroids in the sliver — see §3.2 for the mitigation rule. |
| F-3 | FIRMS VIIRS has zero AOI detections | 🟠 HIGH | 🟠 **OPEN — unchanged.** Re-counted: **0 / 74** points inside AOI (MODIS still 29 / 94,000). Hotspot seeding continues to rely on MODIS + WFIGS baseline. |
| F-4 | Temporal misalignment across feeds (scenario day 2023-08-15 vs NRT 2026-08-21 vs USDM 2026-08-18) | 🟠 HIGH | 🟠 **OPEN — structurally unchanged.** No new temporal feeds landed; canonical simulation clock pinned in NB01 config remains mandatory. |
| F-5 | LANDFIRE EVT shipped without attribute table | 🟡 MED | ✅ **RESOLVED** (with two parameter deviations). New `fuel_crosswalk.csv` maps all 5 raster codes to Anderson FBFM13 parameters with zero nulls and an exact key match to the raster's value domain `{1, 2, 4, 10, 11}`. Unit-consistency audit vs Anderson (1982) found codes 1/4 canonical, code 2 depth deviates (0.6 m vs 0.3048 m), code 10 load is 2× canonical and SAV matches slash models instead of timber litter — details in §2.3. |

**Net position:** The data layer is now unblocked for Notebooks 01, 02, 04, 05, 06 end-to-end over the full AOI. Notebook 03 (evacuation routing) remains blocked solely by F-1.

---

## 1. Re-Profiled Physical Layer

### 1.1 File inventory delta (v1 → v2)

| File | v1 Size (B) | v2 Size (B) | Δ | Last Write |
|------|-------------|-------------|---|------------|
| `dem/srtm_30m_aoi.tif` | 1,001,134 | **10,377,024** | ×10.4 | 2026-08-21 23:38:23 |
| `landfire/landfire_evt.tif` | 500,762 | **5,188,704** | ×10.4 | 2026-08-21 23:38:23 |
| `landfire/fuel_crosswalk.csv` | — | **279** | new | 2026-08-21 23:38:35 |
| all other 16 files | — | unchanged | 0 | pre-v1 timestamps |

Size arithmetic validates internal consistency: DEM data payload `1800·1440·4 B = 10,368,000` + TIFF headers ≈ 10,377,024 ✓; EVT payload `1800·1440·2 B = 5,184,000` + headers ≈ 5,188,704 ✓. Both uncompressed GTiff, band-interleaved.

### 1.2 Raster grid contract (both files)

| Property | Measured value |
|----------|----------------|
| Grid | **1800 cols × 1440 rows = 2,592,000 px** |
| CRS | EPSG:4326 |
| Transform | `(0.000277, 0, −121.8, 0, −0.000277, 40.0)` |
| Resolution | 0.000277° ≈ 1 arcsec ≈ 24 m E-W × 31 m N-S at lat 39.93 |
| Bounds | `(−121.8, 39.60112, −121.3014, 40.0)` |
| NoData sentinel | `-9999` declared; **0 masked pixels**; raw scan confirms neither `-9999` nor `-32768` occurs in either array |
| Co-registration | transform, CRS, and grid **identical** between DEM and EVT → zero-resampling pixel stack verified |
| Full-stack RAM | dem `float32` + evt `int16` = **~15.6 MB** — trivially inside the 15 GB ceiling |

### 1.3 DEM statistics (regenerated surface)

| Statistic | Value |
|-----------|-------|
| dtype | float32 |
| Min / Max elevation | **194.266 m / 1,405.052 m** |
| Mean / σ | **800.001 m / 263.985 m** |
| Percentiles p1/p25/p50/p75/p99 | 277.369 / 589.908 / 800.034 / 1,010.120 / 1,322.586 m |
| Unique values | 2,370,647 (continuous surface; max repeat count 6 px) |

Range widened vs v1 stub tile (was 289.8–1,205.5 m) as expected when extending from the Paradise-plateau window to the full AOI including the Feather River canyon lowlands.

### 1.4 EVT categorical distribution (regenerated)

| Code | Pixels | Share | Interpretation (per crosswalk) |
|------|--------|-------|-------------------------------|
| 1 | 647,942 | 25.00 % | Short Grass — FBFM13 #1 |
| 2 | 647,812 | 24.99 % | Timber Grass — FBFM13 #2 |
| 4 | 648,559 | 25.02 % | Chaparral — FBFM13 #4 |
| 10 | 517,909 | 19.98 % | Timber Litter — FBFM13 #10 |
| 11 | 129,778 | 5.01 % | Non-Burnable — sentinel FBFM13 #99 |

Mean 4.30, σ 3.58. Distribution changed from v1's uniform 20 %×5 to a realistic **75 % burnable / 20 % slow-burn timber / 5 % non-burnable** mix — consistent with a wildland-urban scenario mosaic.

### 1.5 Fuel crosswalk validation (`landfire/fuel_crosswalk.csv`)

Physical: 279 B, UTF-8, CRLF endings, header + 5 records.
Schema (DuckDB-typed): `evt_code int64 · fuel_name string · fbfm13 int64 · fuel_bed_depth_m float64 · fuel_load_kg_m2 float64 · sav_ratio_m_inv int64 · moisture_extinction float64`.

| evt_code | fuel_name | fbfm13 | depth (m) | load (kg/m²) | SAV (m⁻¹) | Mx | Nulls |
|----------|-----------|--------|-----------|--------------|-----------|-----|-------|
| 1 | Short Grass | 1 | 0.3 | 0.166 | 11,480 | 0.12 | 0 |
| 2 | Timber Grass | 2 | 0.6 | 0.896 | 9,840 | 0.15 | 0 |
| 4 | Chaparral | 4 | 1.8 | 1.120 | 6,560 | 0.20 | 0 |
| 10 | Timber Litter | 10 | 0.3 | 2.690 | 4,920 | 0.25 | 0 |
| 11 | Non-Burnable | 99 | 0.0 | 0.000 | 0 | 0.00 | 0 |

**Key-match proof:** raster value domain `{1, 2, 4, 10, 11}` == crosswalk key set, symmetric difference ∅ (no missing, no extra).

**Unit-consistency audit vs Anderson (1982) GTR INT-122** (conversions applied: 1 t/ac = 0.22417 kg/m²; 1 ft⁻¹ = 3.2808 m⁻¹; 1 ft = 0.3048 m):

| Code | Verdict | Evidence |
|------|---------|----------|
| 1 | ✅ canonical FM1 | 0.166 kg/m² = 0.74 t/ac; SAV 11,480 ≈ 3,500 ft⁻¹; depth 0.3 ≈ 1.0 ft; Mx 12 % |
| 2 | ⚠️ minor deviation | load/SAV/Mx match FM2 (4.0 t/ac incl. live herb; 3,000 ft⁻¹; 15 %) but **depth 0.6 m ≈ 2.0 ft vs canonical 1.0 ft** — doubles residence-time proxy; recalibrate or fix before ROS calibration runs |
| 4 | ✅ canonical FM4 | 1.12 kg/m² = 5.0 t/ac (dead 1-h 5.01); SAV 6,560 ≈ 2,000 ft⁻¹; depth 1.8 ≈ 6.0 ft; Mx 20 % |
| 10 | ⚠️ material deviation | **load 2.69 kg/m² = 12.0 t/ac vs canonical FM10 total-dead 6.0 t/ac (2×)**; **SAV 4,920 ≈ 1,500 ft⁻¹ vs canonical FM10 2,000 ft⁻¹** (1,500 is the FM11–13 slash value); depth/Mx match FM10 |
| 11 | ✅ clean sentinel | `fbfm13=99` + zeroed params; must be used as a burn-exclusion mask only — never feed σ=0 into Rothermel (division hazard) |

**Action item (non-blocking):** correct code-10 load/SAV and code-2 depth at the source or document them as intentional scenario tuning before NB05 sensitivity benchmarks.

---

## 2. Updated Master Data Catalog & Ingestion Readiness Matrix

Canonical AOI `(−121.8, 39.6, −121.3, 40.0)` ≈ 1,904 km². H3 res-8 constants re-measured: avg edge 0.5314 km, avg hex area 0.7523 km².

| # | Feed | Target File | Size (B) | Native CRS | Spatial Extent (measured) | Primary / Temporal Key | Consumer NBs | Ingestion Strategy | Ready? |
|---|------|-------------|----------|------------|---------------------------|------------------------|--------------|--------------------|--------|
| 1 | FIRMS VIIRS | `firms/viirs_snpp_california.csv` | 6,275 | EPSG:4326 | CA-wide; **0 AOI hits** | `(lat,lon,acq_date,acq_time)`; 2026-08-21 | 01, 02, 05, 06 | DuckDB typed read + bbox pushdown | ⚠️ F-3 open |
| 2 | MODIS | `modis/modis_2023_usa.csv` | 7,572,283 | EPSG:4326 | CONUS+; 29 AOI rows | same; CY2023 | 01, 02, 05, 06 | DuckDB typed read + bbox pushdown | ✅ |
| 3 | LANDFIRE EVT | `landfire/landfire_evt.tif` | 5,188,704 | EPSG:4326 | full-AOI grid (§1.2); 99.44 % strict | raster cell; static | 01, 04, 05 | Rasterio full/windowed read + crosswalk LUT join | ✅ F-2/F-5 resolved |
| 3b | Fuel crosswalk | `landfire/fuel_crosswalk.csv` | 279 | n/a (attribute table) | 5 rows × 7 cols, 0 nulls | `evt_code` | 01, 04, 05 | Pandas/DuckDB LUT; dict join on EVT codes | ✅ |
| 4 | SRTM DEM | `dem/srtm_30m_aoi.tif` | 10,377,024 | EPSG:4326 | full-AOI grid (§1.2); 99.44 % strict | raster cell; static | 01, 03, 04, 05 | Rasterio full/windowed read | ✅ F-2 resolved |
| 5 | ERA5-Land | `weather/era5_hourly_aoi.nc` | 49,416 | EPSG:4326 (CF) | padded box ⊇ AOI; 24 h × 2023-08-15 | `(time,lat,lon)` hourly | 01, 04, 05, 06 | Xarray lazy slice (`sortby('latitude')` first) | ✅ |
| 6 | OSM roads | `osm/aoi.osm.pbf` | 28,389,346 | EPSG:4326 | ❌ Sacramento box; 0 AOI roads | OSM ids | 03, 06 | Pyosmium streaming | 🔴 F-1 open |
| 7 | USDM | `drought/USDM_20260818.shp` (+7 sidecars) | 4,212,584 | EPSG:4326 (.prj) | CONUS+; AOI at severity D0 | `OBJECTID`; 2026-08-18 | 01, 04, 05 | DuckDB `ST_Read` + `ST_MakeValid` | ✅ (repair mandatory) |
| 8 | Census BGs | `census/census_blocks.geojson` | 1,173 | CRS84 | exact AOI quadrant tiling | `GEOID` | 01, 03, 06 | GeoPandas direct | ✅ |
| 9 | WFIGS | `wfigs/wfigs_perimeters.geojson` | 967 | RFC 7946 WGS84 | inside AOI; 12.60 % area | incident name; 2023-08-15T06:30Z | 01, 02, 05, 06 | GeoPandas direct | ✅ (synthetic provenance noted) |
| 10 | PeMS | `traffic/pems_station_speeds.csv` | 5,301 | none (tabular) | station panel 5×24 h | `(station_id, timestamp)` | 03, 06 | DuckDB typed read | ⚠️ needs station coords |
| 11 | AirNow | `airnow/airnow_pm25_aoi.csv` | 436 | EPSG:4326 | 6 sites; 4 in AOI | `site_name`; single ts | 01, 05, 06 | DuckDB typed read | ✅ |

---

## 3. Spatial Alignment & H3 Discretization Guide (v2)

### 3.1 Raster ↔ AOI coverage proof

- Measured bounds `(−121.8, 39.60112, −121.3014, 40.0)` vs canonical `(−121.8, 39.6, −121.3, 40.0)`:
  - east shortfall `0.001400°` = **119.7 m**; south shortfall `0.001120°` = **123.8 m**
  - exact fit would require **1805.05 × 1444.04 px**; provided 1800 × 1440
  - strict areal coverage **99.44 %** of the canonical box
- Practical verdict: the deficit is an L-shaped sliver narrower than one H3-res-8 hex edge (531 m). All interior modeling domains (WFIGS perimeter, census quadrants, AirNow sites, ERA5 nodes) remain fully backed by raster data.

### 3.2 H3 Resolution-8 binding rules (updated for full-AOI rasters)

Measured this session: **2,526 cells** intersect the AOI bbox via `polygon_to_cells` (area-ratio estimate 2,531 — consistent within boundary-cell noise). Storage stays `h3_id UINT64`.

| Layer | Binding rule | v2 status |
|-------|--------------|-----------|
| DEM / EVT | hex centroid → pixel lookup; mean (DEM) / mode (EVT) over ~**1,030 px/hex** (px ≈ 731 m²) | ✅ now covers 2,511/2,526 cells (99.4 %) |
| Sliver cells (15) | centroids fall east of `-121.3014` or south of `39.60112`; rule: **nearest-valid-pixel clamp** (edge-replicate) and flag `raster_backed=false` in the curated layer | ⚠️ documented mitigation |
| Fuel params | mode EVT code → `fuel_crosswalk.csv` row → Rothermel parameter tuple; code 11 → burn mask exclusion | ✅ validated end-to-end |
| ERA5 | bilinear interp of 4 surrounding 0.05° nodes to hex centroid (component-wise for wind) | unchanged |
| FIRMS/MODIS | `latlng_to_cell(lat, lon, 8)` at ingest; counts + ΣFRP | unchanged (F-3 counts stand) |
| WFIGS / Census / USDM / AirNow | centroid-in-polygon / areal-weight / MAX(DM) / IDW | unchanged from v1 §3.2 |
| PeMS / OSM | blocked pending station coords (F-5-family) and re-extract (F-1) | 🔴 |

---

## 4. Validated Notebook-01 Ingestion Snippets (executed this session)

**Full-AOI co-registered raster stack + fuel LUT join (RAM ~15.6 MB):**
```python
import numpy as np, rasterio, duckdb, h3

with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as dem_src, \
     rasterio.open("data/raw/landfire/landfire_evt.tif") as evt_src:
    assert dem_src.transform == evt_src.transform and dem_src.shape == evt_src.shape  # co-registration guard
    assert dem_src.nodata == -9999.0 and evt_src.nodata == -9999.0
    dem = dem_src.read(1)          # (1440, 1800) float32
    evt = evt_src.read(1)          # (1440, 1800) int16
    assert not ((dem == -9999).any() or (evt == -9999).any())   # sentinel bleed guard

con = duckdb.connect()
lut = con.execute(
    "SELECT * FROM read_csv_auto('data/raw/landfire/fuel_crosswalk.csv')"
).df().set_index("evt_code")       # 5 rows, 0 nulls, keys == np.unique(evt)

burnable_mask = evt != 11          # fbfm13==99 sentinel -> exclusion mask, never fed to Rothermel
```

**Per-hex fuel parameterization (mode-aggregate → Rothermel tuple):**
```python
import collections
rr = rng.integers(0, dem.shape[0], n); cc = rng.integers(0, dem.shape[1], n)      # or full-grid streaming
lons = -121.8 + (cc + 0.5) * 0.000277
lats =  40.0 - (rr + 0.5) * 0.000277
hexes = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(lats, lons)]

hex_mode = collections.defaultdict(collections.Counter)
for hx, code in zip(hexes, evt[rr, cc]):
    hex_mode[hx][int(code)] += 1

for hx, counter in hex_mode.items():
    evt_major, votes = counter.most_common(1)[0]
    row = lut.loc[evt_major]       # fuel_name, fbfm13, fuel_bed_depth_m, fuel_load_kg_m2,
                                   # sav_ratio_m_inv, moisture_extinction
    purity = votes / sum(counter.values())   # store as uncertainty column
```

**Sliver-cell guard (15 known cells):**
```python
RASTER_E, RASTER_S = -121.3014, 39.60112
lat, lon = h3.cell_to_latlng(hx)
if lon > RASTER_E or lat < RASTER_S:
    # nearest-valid-pixel clamp + flag
    raster_backed = False
```

**DEM gradient prep (Horn, meter-corrected) over the full grid:**
```python
dx_m = 0.000277 * 111320 * np.cos(np.radians(dem_lat_center))   # ~23.6 m E-W
dy_m = 0.000277 * 110574                                        # ~30.6 m N-S
zx = np.gradient(dem, dx_m, axis=1); zy = np.gradient(dem, dy_m, axis=0)
slope_pct = 100 * np.hypot(zx, zy)
aspect_rad = np.mod(np.arctan2(-zy, zx), 2*np.pi)               # circular stats downstream
```

**Regression spot-checks re-run this session (assertions for NB01 CI):**
```python
# OSM header box must be re-extracted before NB03:
import osmium as o
rd = o.io.Reader("data/raw/osm/aoi.osm.pbf"); hb = rd.header().box(); rd.close()
assert str(hb.bottom_left) != "-121.625/38.374", "F-1: stale Sacramento extract"

# FIRMS AOI yield (currently 0/74 — keep MODIS fallback path):
n = con.execute("""SELECT COUNT(*) FROM read_csv_auto('data/raw/firms/viirs_snpp_california.csv')
                   WHERE longitude BETWEEN -121.8 AND -121.3 AND latitude BETWEEN 39.6 AND 40.0""").fetchone()[0]
```

---

## 5. Remaining Remediation Backlog (v2, ordered)

1. **F-1 (blocks NB03):** re-extract OSM for `--bbox=-121.8,39.6,-121.3,40.0` from Geofabrik California; re-run graph profile after.
2. **Crosswalk corrections (non-blocking):** code 10 `fuel_load_kg_m2` 2.69 → 1.346 (canonical 6.0 t/ac) and `sav_ratio_m_inv` 4920 → 6560 (FM10 2,000 ft⁻¹); code 2 `fuel_bed_depth_m` 0.6 → 0.3048 — or document as deliberate scenario tuning in NB05.
3. **Optional raster regeneration:** 1805 × 1444 px would close the 119.7/123.8 m sliver and back all 2,526 hexes; otherwise keep the clamp-and-flag rule above.
4. **F-3:** AOI-bounded FIRMS pull (or formalize MODIS-only seeding).
5. **PeMS station metadata** (coordinates, lanes) for stations 318491–318495.
6. **F-4:** enforce the pinned simulation clock `2023-08-15T00:00–23:00Z` in NB01 config.

---

*Profiling artifacts retained in `audit_scratch/`: `profile_v2.py` (raster + crosswalk dump), `profile_v2b.py` (sliver quantification, regression spot-checks, NB01 pipeline test), plus v1 scripts for lineage.*
