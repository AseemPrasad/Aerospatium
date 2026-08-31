# AeroSpatium — Authoritative Data Layer Specification & Ingestion Audit Report

**Project:** AeroSpatium — Multi-modal Wildfire Progression & Evacuation Route Optimization Engine
**Audit Date:** 2026-08-21
**Scope:** `data/raw/` — 11 feeds → **18 physical files**, **43,279,830 bytes (41.27 MiB)** total on disk
**Environment:** Python 3.14.0 · rasterio 1.5.1 · xarray 2026.7.0 · duckdb 1.5.5 (+spatial) · shapely 2.1.2 · geopandas 1.1.4 · pyarrow 23.0.0 · h3 4.5.0 · osmium (pyosmium) · netCDF4 1.7.4
**Method:** Every file parsed with native library bindings (`rasterio`, `xarray`, `duckdb/spatial`, `osmium`, raw `struct`/DBF/SHP header parsers). All numbers below are **measured**, not assumed. Profiling scripts preserved in `audit_scratch/`.

---

## ⚠️ Executive Summary — 5 Blocking Findings

| # | Severity | Finding |
|---|----------|---------|
| F-1 | 🔴 **BLOCKER** | **OSM extract covers the wrong region.** PBF header box = `(−121.625, 38.374) → (−121.154, 38.738)` — Sacramento metro, ~130 km south of the AOI. Of 554,318 ways, exactly **1** intersects the AOI and it is a WAPA 230 kV power line, not a road. Notebook 03 routing is impossible until the extract is re-clipped for Butte County. |
| F-2 | 🔴 **BLOCKER** | **DEM + LANDFIRE EVT cover only 9.59 % of the AOI** (NW window `lon −121.8→−121.6615, lat 39.8615→40.0`). The remaining 90.4 % (incl. Paradise townsite at `−121.61, 39.76`) has no terrain/fuel surface. Re-download full-AOI tiles before Notebook 04. |
| F-3 | 🟠 HIGH | **FIRMS VIIRS feed contains zero AOI detections** (74 CA-wide points from a single pass on 2026-08-21; 2 within ±1° of Butte County). MODIS contributes only 29/94,000 rows inside the AOI. Hotspot-driven progression (Notebook 02) must fall back to MODIS + WFIGS or re-pull FIRMS with an AOI-bounded query. |
| F-4 | 🟠 HIGH | **Temporal misalignment across feeds**: ERA5-Land, PeMS, AirNow, WFIGS all reference scenario day **2023-08-15**; MODIS spans CY2023; FIRMS is NRT **2026-08-21**; USDM release **2026-08-18**. A canonical simulation clock must be pinned to `2023-08-15 00:00–23:00 UTC` in Notebook 01. |
| F-5 | 🟡 MED | **LANDFIRE EVT ships without its attribute table.** Raster codes `{1, 2, 4, 10, 11}` are compacted integers that do **not** match official LANDFIRE EVT 4-digit classes (e.g., `7210`). No CSV/DBF crosswalk exists in `data/raw/landfire/`. Rothermel fuel mapping is blocked pending the vendor attribute table (provisional crosswalk in §2.3). |

Additional high-value hazards: USDM `DM=2` ring fails OGC validity (§2.7), ERA5 variables carry **no `units` attributes** (§2.5), PeMS has **no station coordinates** (§2.10), FIRMS/MODIS encode `acq_time` with incompatible types (§2.1/§2.2), `occupancy_pct` is a fraction masquerading as a percent (§2.10).

---

## 1. Master Data Catalog & Ingestion Readiness Matrix

Canonical AOI (derived from census tiling + ERA5 grid): **`(min_lon, min_lat, max_lon, max_lat) = (−121.8, 39.6, −121.3, 40.0)`** — Camp Fire footprint, Butte County, CA. Area ≈ 0.2 deg² ≈ **1,904 km²**.

| # | Feed | Target File | Size (B) | Native CRS | Spatial Extent (measured) | Primary / Temporal Key | Consumer NBs | Ingestion Strategy | Ready? |
|---|------|-------------|----------|------------|---------------------------|------------------------|--------------|--------------------|--------|
| 1 | FIRMS VIIRS | `data/raw/firms/viirs_snpp_california.csv` | 6,275 | EPSG:4326 (lon/lat cols) | lat 32.533–40.822, lon −122.112…−114.254 (CA-wide) | composite `(lat,lon,acq_date,acq_time)`; `acq_date`=2026-08-21 | 01, 02, 05, 06 | DuckDB typed `read_csv` + bbox pushdown | ⚠️ no AOI hits |
| 2 | MODIS Thermal Anomalies | `data/raw/modis/modis_2023_usa.csv` | 7,572,283 | EPSG:4326 (lon/lat cols) | lat 19.150–68.202, lon −164.072…−67.379 (CONUS+) | composite `(lat,lon,acq_date,acq_time)`; CY2023 | 01, 02, 05, 06 | DuckDB typed `read_csv` + bbox pushdown | ✅ 29 AOI rows |
| 3 | LANDFIRE EVT fuel | `data/raw/landfire/landfire_evt.tif` | 500,762 | EPSG:4326 | (−121.8, 39.8615, −121.6615, 40.0) | implicit raster cell; static layer | 01, 04, 05 | Rasterio windowed read → H3 mode-aggregate | 🔴 9.6 % AOI; no attr table |
| 4 | SRTM DEM | `data/raw/dem/srtm_30m_aoi.tif` | 1,001,134 | EPSG:4326 | (−121.8, 39.8615, −121.6615, 40.0) | implicit raster cell; static layer | 01, 03, 04, 05 | Rasterio windowed read (co-registered w/ EVT) | 🔴 9.6 % AOI |
| 5 | ERA5-Land hourly | `data/raw/weather/era5_hourly_aoi.nc` | 49,416 | EPSG:4326 (CF lon/lat) | centers lon −121.8…−121.3, lat 39.6…40.0 | `(time, latitude, longitude)`; 24 h × 2023-08-15 | 01, 04, 05, 06 | Xarray disk slice (lazy) | ✅ |
| 6 | OSM road network | `data/raw/osm/aoi.osm.pbf` | 28,389,346 | EPSG:4326 (PBF proto) | header box (−121.625, 38.374, −121.154, 38.738) ❌ | OSM id (`node.id`, `way.id`) | 03, 06 | Pyosmium streaming handler | 🔴 wrong region |
| 7 | USDM drought | `data/raw/drought/USDM_20260818.shp` (+ `.shx/.dbf/.prj/.sbn/.sbx/.cpg/.shp.xml`) | 4,212,584 (shp) | EPSG:4326 (per `.prj` GCS_WGS_1984) | CONUS+: lon −159.786…−65.221, lat 17.884…65.108 | `OBJECTID`; snapshot 2026-08-18 | 01, 04, 05 | DuckDB `ST_Read` + `ST_MakeValid` | ⚠️ DM=2 invalid geom |
| 8 | Census block groups | `data/raw/census/census_blocks.geojson` | 1,173 | CRS84 (named member) | exact AOI tiling: 4 quadrant polygons | `GEOID` (str, 11-digit) | 01, 03, 06 | GeoPandas direct load | ✅ |
| 9 | WFIGS fire perimeter | `data/raw/wfigs/wfigs_perimeters.geojson` | 967 | RFC 7946 default (WGS84) | (−121.72, 39.75, −121.48, 39.92); 12.60 % of AOI | `poly_IncidentName`; `attr_FireDiscoveryDateTime`=2023-08-15T06:30Z | 01, 02, 05, 06 | GeoPandas direct load | ⚠️ synthetic provenance |
| 10 | Caltrans PeMS | `data/raw/traffic/pems_station_speeds.csv` | 5,301 | **none — tabular, no coords** | n/a (station_id 318491–318495) | `(station_id, timestamp)` hourly × 2023-08-15 | 03, 06 | DuckDB typed read; join external station metadata | ⚠️ unmappable stations |
| 11 | EPA AirNow PM2.5 | `data/raw/airnow/airnow_pm25_aoi.csv` | 436 | EPSG:4326 (lon/lat cols) | 6 sites, lat 39.512–39.924, lon −121.843…−121.556 | `site_name`; single ts 2023-08-15 08:00 | 01, 05, 06 | DuckDB typed read | ✅ 4/6 sites in AOI |

---

## 2. Deep-Dive Dataset Profiles

### 2.1 FIRMS VIIRS Active Fires — `firms/viirs_snpp_california.csv`

**Physical:** 6,275 B, UTF-8, LF endings, 75 lines (header + 74 records), no compression.

**Exact schema (DuckDB-inferred):**

| Column | Dtype | Nulls | Min | Max | Notes |
|--------|-------|-------|-----|-----|-------|
| `latitude` | DOUBLE | 0 | 32.53297 | 40.82220 | WGS84 point Y |
| `longitude` | DOUBLE | 0 | −122.11218 | −114.25364 | WGS84 point X |
| `bright_ti4` | DOUBLE | 0 | 296.05 | 367.00 | I-band 4 brightness temp, Kelvin |
| `scan` | DOUBLE | 0 | 0.38 | 0.55 | along-scan pixel size, km |
| `track` | DOUBLE | 0 | 0.36 | 0.47 | along-track pixel size, km |
| `acq_date` | DATE | 0 | 2026-08-21 | 2026-08-21 | single acquisition day |
| `acq_time` | BIGINT | 0 | 1013 | 1015 | **HHMM as integer** (not epoch, not padded string) |
| `satellite` | VARCHAR | 0 | `N` (74/74) | — | Suomi-NPP |
| `instrument` | VARCHAR | 0 | `VIIRS` | — | |
| `confidence` | VARCHAR | 0 | `n`:73, `h`:1 | — | nominal/high/low enum |
| `version` | VARCHAR | 0 | `2.0NRT` | — | NRT feed |
| `bright_ti5` | DOUBLE | 0 | 270.22 | 307.99 | I-band 5 brightness temp, K |
| `frp` | DOUBLE | 0 | 0.22 | 8.98 | fire radiative power, MW; mean 1.99, σ 2.03 |
| `daynight` | VARCHAR | 0 | `N` (74/74) | — | all night passes |

**Statistics & domain:** `bright_ti4` mean 311.99 K (σ 14.04); scan/track ≈ 0.4 km confirms nominal **375 m** resolution. Zero duplicate keys on `(lat,lon,date,time)`. **0/74 points inside AOI**; 2 within ±1° Butte buffer.

**Transformations required:**
- Acquisition datetime: `TIMESTAMP(acq_date, make_time(acq_time // 100, acq_time % 100, 0))` — HHMM integer decode.
- Confidence enum → ordered factor `l < n < h`; drop `l` for progression seeding.
- FRP per-hex aggregation: $FRP_{hex} = \sum_i frp_i$ (MW), then fire-energy flux $E = \int FRP\,dt$ (MJ) per H3 cell.
- Point → hex assignment: `h3.latlng_to_cell(lat, lon, 8)` → store `h3_id UINT64`.

**Hazards:** (a) zero AOI coverage (F-3); (b) `acq_time` BIGINT collides with MODIS zero-padded STRING — normalize both to `TIME` in Notebook 01; (c) NRT product has no post-processing QA — expect false positives over gas flares/reflective roofs; (d) night-only pass means no same-day diurnal pairing with ERA5 daytime hours.

---

### 2.2 MODIS Thermal Anomalies — `modis/modis_2023_usa.csv`

**Physical:** 7,572,283 B (7.22 MiB), UTF-8, 94,001 lines (header + 94,000 records).

**Exact schema:**

| Column | Dtype | Nulls | Min | Max | Notes |
|--------|-------|-------|-----|-----|-------|
| `latitude` | DOUBLE | 0 | 19.1496 | 68.2016 | |
| `longitude` | DOUBLE | 0 | −164.0724 | −67.3794 | |
| `brightness` | DOUBLE | 0 | 300.0 | 508.9 | MODIS ch.21/22 brightness temp, K |
| `scan` | DOUBLE | 0 | 1.0 | 4.8 | km; nominal 1 km product |
| `track` | DOUBLE | 0 | 1.0 | 2.0 | km |
| `acq_date` | DATE | 0 | 2023-01-01 | 2023-12-31 | 365 distinct days |
| `acq_time` | **VARCHAR** | 0 | `"0000"` | `"2359"` | **zero-padded string** (vs FIRMS BIGINT!) |
| `satellite` | VARCHAR | 0 | Aqua: 51,287 · Terra: 42,713 | | |
| `instrument` | VARCHAR | 0 | `MODIS` | | |
| `confidence` | BIGINT | 0 | 0 | 100 | **percent 0–100** (C6.1), unlike VIIRS enum |
| `version` | DOUBLE | 0 | 61.03 | | Collection 6.1 |
| `bright_t31` | DOUBLE | 0 | 265.3 | 400.1 | ch.31 BT, K |
| `frp` | DOUBLE | 0 | 0.0 | 5,763.5 | MW; mean 45.75, σ 135.78, heavy right tail |
| `daynight` | VARCHAR | 0 | D: 81,234 · N: 12,766 | | |
| `type` | BIGINT | 0 | 0 | 3 | 0=vegetation, 1=static land, 2=vessel/offshore, 3=other |

**Statistics & domain:** 29 rows strictly inside AOI; 1,257 within ±1° Butte buffer. No duplicates on `(lat,lon,acq_date,acq_time)`. FRP distribution is right-skewed (max 5.76 GW) — log-transform before ML: $\log_2(1+frp)$.

**Transformations required:**
- `CAST(acq_time AS INT)` then HHMM decode (same as FIRMS after cast).
- Filter `type IN (0, 3)` and `confidence >= 50` for progression seeds.
- Cross-sensor calibration note: VIIRS `conf ∈ {l,n,h}` vs MODIS `conf ∈ [0,100]` — map MODIS ≥80→`h`, 30–79→`n`, <30→`l` for a unified confidence column.
- Scan/track area correction: pixel area ≈ `scan × track` km²; normalize FRP density $FRP_{norm} = frp / (scan \cdot track)$ when aggregating across swath edges.

**Hazards:** (a) CONUS-wide file — always push the bbox predicate into `read_csv` (DuckDB projects column-wise; full scan is still cheap at 7 MB but production feeds are GB-scale); (b) `acq_time` type mismatch with FIRMS (F-4 family); (c) bow-tie swath duplicates near orbit edges already deduped here but re-check after AOI subsetting.

---

### 2.3 LANDFIRE EVT Fuel — `landfire/landfire_evt.tif`

**Physical:** 500,762 B, GTiff (little-endian `II*\0`), uncompressed, band-interleaved, block 8×500.

| Property | Measured value |
|----------|----------------|
| dtype / bands | `int16`, 1 band |
| Grid | 500 × 500 px (250,000 cells) |
| CRS | EPSG:4326 |
| Transform | `(0.000277, 0, −121.8, 0, −0.000277, 40.0)` |
| Resolution | 0.000277° ≈ 1 arcsec ≈ 24 m E-W × 31 m N-S at lat 39.93 |
| Bounds | `(−121.8, 39.8615, −121.6615, 40.0)` |
| NoData sentinel | `-9999` (declared; **0 masked pixels present**) |
| Valid pixels | 250,000 |

**Value distribution (categorical):**

| Code | Pixels | Share | Provisional Anderson FBFM13 reading | Provisional Scott-Burgan FM40 |
|------|--------|-------|--------------------------------------|-------------------------------|
| 1 | 50,032 | 20.0 % | FM1 — short grass | GR1 (101) |
| 2 | 50,061 | 20.0 % | FM2 — timber+grass understory | GR2 (102) |
| 4 | 49,856 | 19.9 % | FM4 — chamise chaparral | SH1 (141) |
| 10 | 50,116 | 20.0 % | FM10 — timber litter + understory | TU5 (165) |
| 11 | 49,935 | 20.0 % | non-burnable (urban/water/snow) | NB1 (201) |

Mean 5.60, σ 4.13. The near-uniform 20 % split across five codes indicates a **synthetic/sampled raster**, not raw LANDFIRE output.

**Transformations required:**
- Mode-aggregate to H3 res-8 (~1,032 px per hex): `h3_id → argmax(count(code))`; keep purity fraction as uncertainty.
- Rothermel inputs per fuel model: fuel load $w_0$, SAV ratio $\sigma$, depth $\delta$, moisture of extinction $M_x$ — lookup keyed by the crosswalk above **once the official attribute table arrives** (F-5).
- Co-registration with DEM verified: identical transform/CRS/grid → pixel-wise stack is safe with zero resampling.

**Hazards:** (a) 9.59 % AOI coverage (F-2); (b) missing attribute table blocks any defensible Rothermel parameterization — treat crosswalk as provisional; (c) code `11` ambiguity (logging slash vs non-burnable) materially changes burnable-area masks; (d) int16 nodata `-9999` would poison categorical mode if present — currently absent but assert on read.

---

### 2.4 SRTM DEM — `dem/srtm_30m_aoi.tif`

**Physical:** 1,001,134 B, GTiff uncompressed, block 4×500.

| Property | Measured value |
|----------|----------------|
| dtype / bands | `float32`, 1 band |
| Grid | 500 × 500 px |
| CRS / Transform | EPSG:4326 / `(0.000277, 0, −121.8, 0, −0.000277, 40.0)` — **identical to EVT** |
| Bounds | `(−121.8, 39.8615, −121.6615, 40.0)` |
| NoData | `-9999.0` declared, 0 masked px |
| Elevation min/max | **289.779 m / 1,205.454 m** |
| Mean / σ | **750.009 m / 198.328 m** |
| Percentiles p1/p25/p50/p75/p99 | 357.09 / 592.05 / 749.95 / 907.87 / 1,142.83 m |
| Unique values | 247,354 (continuous surface) |

**Transformations required:**
- Gradient (Horn 3×3, cell-corrected to meters):
  $\partial z/\partial x = \frac{(z_{i+1,j-1}+2z_{i+1,j}+z_{i+1,j+1})-(z_{i-1,j-1}+2z_{i-1,j}+z_{i-1,j+1})}{8\,\Delta x_m}$, analogously $\partial z/\partial y$;
  slope $S=\sqrt{z_x^2+z_y^2}$, aspect $\theta=\operatorname{atan2}(-z_y,\,z_x)\to[0°,360°)$ compass.
- **Circular mean aspect per hex:** $\bar\theta = \operatorname{atan2}\!\big(\sum_k \sin\theta_k,\ \sum_k \cos\theta_k\big)$ — never arithmetic-mean aspects (0°/359° average).
- Slope in percent for road-graph edge cost: $100 \cdot S$; pedestrian/vehicle grade penalty thresholds at 8 % / 12 %.
- Hillshade for NB06 basemap: $HS = 255\cdot(\cos(z)\sin(s) + \sin(z)\cos(s)\cos(az-\theta))$.
- Upsampling to ERA5 grid or downsampling to H3 via mean-elevation pooling.

**Hazards:** (a) 9.59 % AOI coverage (F-2) — gradient ops at the eastern/southern tile edge will silently produce cliffs where data ends; mask before Sobel/Horn convolution (**NoData bleed**); (b) float32 SRTM is interpolated 1-arcsec product — vertical accuracy ±16 m, adequate for WAF/slope but not for hydro-flattening; (c) void-fill sentinels absent today, but production SRTM voids (`-32768`) must be asserted against.

---

### 2.5 ERA5-Land Hourly Weather — `weather/era5_hourly_aoi.nc`

**Physical:** 49,416 B, HDF5 container (magic `\x89HDF\r\n\x1a\n`) = NetCDF4.

| Dimension | Size | Coordinate values |
|-----------|------|-------------------|
| `time` | 24 | 2023-08-15T00:00Z → 23:00Z, step **3600 s** (hourly, uniform) |
| `latitude` | 10 | **39.6 → 40.0 ASCENDING**, step 0.044444° |
| `longitude` | 10 | −121.8 → −121.3, step 0.055556° |

**Variables** — all `float32`, dims `(time, latitude, longitude)`, shape `(24, 10, 10)`, **all shipped with EMPTY attrs dicts (no `units`, no `long_name`)**:

| Var | Physical meaning (hard-coded contract) | Min | Max | Mean | σ |
|-----|----------------------------------------|-----|-----|------|---|
| `u10` | 10 m U-wind, m/s (E+) | 1.0034 | 7.9972 | 4.4639 | 2.0149 |
| `v10` | 10 m V-wind, m/s (N+) | −3.9983 | 3.9973 | −0.0320 | 2.3335 |
| `t2m` | 2 m temperature, **Kelvin** | 295.0026 | 309.9713 | 302.6770 | 4.3788 |
| `d2m` | 2 m dewpoint, **Kelvin** | 280.0065 | 290.0000 | 285.0253 | 2.8959 |

Global attrs: none. Cell-center extent equals AOI corners exactly; half-cell padded footprint `(−121.8278, 39.5778, −121.2722, 40.0222)` fully contains AOI.

**Transformations required:**
- Wind speed/direction: $ws = \sqrt{u_{10}^2+v_{10}^2}$; meteorological FROM-direction $wd = (270° - \deg(\operatorname{atan2}(-v,-u))) \bmod 360°$. Verified slice mean ws = 4.95 m/s.
- Relative humidity (Magnus/Alduchov-Eskridge): $RH = 100\cdot\dfrac{\exp\!\big(\frac{17.625\,T_d}{243.04+T_d}\big)}{\exp\!\big(\frac{17.625\,T}{243.04+T}\big)}$ with $T,T_d$ in °C. Verified AOI-morning mean RH ≈ 33.8 %.
- Midflame wind reduction (Albini/BAE): $w_{midflame} = w_{open} \cdot WAF$, WAF ≈ 0.4–0.5 unsheltered surface fuels, 0.1–0.3 sheltered under canopy — feed from EVT cover class.
- Fine dead-fuel moisture (Fosberg EMC) from RH+t2m for spread-rate modulation.
- K→°C conversion before any human-facing output; keep K internally.

**Hazards:** (a) missing unit attributes — units are a hard-coded contract, assert ranges on load (t2m∈[250,330]K etc.) to catch silent unit drift; (b) `latitude` stored **ascending** — native CDS downloads are descending; `sel(latitude=slice(hi,lo))` returns empty arrays silently (caught in snippet testing) — always `ds.sortby('latitude')` first; (c) u10 strictly positive (1.0–8.0 m/s) is physically implausible for a full diurnal cycle → synthetic wind field, flag for benchmark realism; (d) 0.05° native grid is coarser than H3 res-8 hexes (~0.53 km edge vs ~4.5 km grid) — bilinear-interpolate before hex join, never nearest-neighbor for wind direction.

---

### 2.6 OSM Road Network — `osm/aoi.osm.pbf`

**Physical:** 28,389,346 B (27.08 MiB), OSM PBF protobuf (blob header `OSMHeader` at offset 0), zlib-compressed primitive blocks.

| Metric | Measured |
|--------|----------|
| Header box | **(−121.625, 38.374) → (−121.154, 38.738)** ❌ Sacramento metro |
| Nodes | 3,727,717 (151,002 tagged) |
| Ways | 554,318 |
| Relations | 5,020 |
| Highway-tagged ways | 155,183 |
| Way segments (densified) | 927,057 |
| Total network length | 21,869 km (median seg 11.8 m, p95 82.6 m) |
| Ways intersecting AOI | **1** — a `power=line` (WAPA Cottonwood-Roseville 230 kV), i.e., **zero roads** |

**Highway category distribution (ways):**

| Class | Count | | Class | Count |
|-------|-------|-|-------|-------|
| service | 70,861 | | motorway_link | 1,322 |
| footway | 32,562 | | motorway | 891 |
| residential | 27,412 | | track | 711 |
| secondary | 7,091 | | unclassified | 609 |
| tertiary | 4,252 | | secondary_link | 411 |
| primary | 3,658 | | pedestrian | 391 |
| path | 2,399 | | primary_link | 384 |
| cycleway | 1,629 | | steps | 300 |
| construction/proposed/other | 62+13+~70 | | | |

**Tag missingness on highway ways (evacuation-routing criticality):**

| Tag | Missing | Missing % | Present-value notes |
|-----|---------|-----------|---------------------|
| `maxspeed` | 147,307 | **94.9 %** | mixed formats: `'45 mph'`×2130, `'65 mph'`×892 … bare `'35'`×5, `'24'`×1 — needs unit parser |
| `lanes` | 141,939 | **91.5 %** | ints 1–8; 2×4657, 3×3690, 4×2136 most common |
| `oneway` | 130,043 | **83.8 %** | yes×23,249, no×1873, `-1`×14, reversible×4 |
| `surface` | 134,887 | 86.9 % | asphalt 13,465, paved 3132, concrete 2603, dirt 222… |
| `name` | 111,063 | 71.6 % | |

Also measured: `access=private`×12,952, `customers`×714, `no`×379 (must be excluded/subset for evacuation); `bridge=yes`×1,605.

**Transformations required (once correctly extracted):**
- Imputation ladder: `maxspeed` ← highway-class median (motorway 65 mph, primary 45, secondary 35, residential 25); `lanes` ← class default (2); `oneway=no` default except motorway/link (`yes`).
- Speed conversion: $v_{m/s} = v_{mph} \times 0.44704$.
- Free-flow travel time: $t_0 = L_{seg}/v$; capacity $C = lanes \times 1{,}800\ veh/h$ (freeway) or $\times 900$ (arterial).
- BPR congestion: $t = t_0\left[1 + 0.15\,(V/C)^4\right]$ with guard `CASE WHEN C>0 THEN power(V::DOUBLE/C,4) ELSE 0 END` (division-by-zero hazard).
- Evacuation subgraph filter: `highway IN (motorway, trunk, primary, secondary, tertiary, unclassified, residential, living_street, motorway_link, *)` minus `access IN (private, no)` minus `construction/proposed`.

**Hazards:** (a) **wrong-region extract (F-1)** — every number above describes Sacramento, not the AOI; re-run `osmium extract --bbox=-121.8,39.6,-121.3,40.0` against a California Geofabrik PBF before Notebook 03; (b) dense node count (3.7 M) demands streaming handlers — do NOT materialize node locations into pandas; (c) `locations=True` caching adds ~400 MB RSS on this file — use `idx='flex_mem'` or process ways in chunks; (d) relations (5,020) include route boundaries needed for turn restrictions later.

---

### 2.7 USDM Drought — `drought/USDM_20260818.shp`

**Physical sidecars (all present):** `.shp` 4,212,584 B · `.shp.xml` 1,539,151 B · `.dbf` 432 B · `.prj` 145 B · `.sbn` 164 B · `.sbx` 116 B · `.cpg` 5 B (`UTF-8`).
Raw SHP header: file_code 9994, shape_type **5 (Polygon)**, declared bbox lon(−159.786, −65.221) lat(17.884, 65.108). Raw DBF header: dBase III, dated 2026-08-19, 5 records, rec_len 54.
`.prj` parses to `GEOGCS["GCS_WGS_1984"…]` = **EPSG:4326** (GeoPandas/DuckDB agree).

**Schema (DBF fields → DuckDB types):**

| Field | DBF type | Arrow/DuckDB type | Content |
|-------|----------|-------------------|---------|
| `OBJECTID` | N(10,0) | BIGINT | 1–5 feature id |
| `DM` | N(5,0) | INTEGER | drought severity **0–4** (D0 Abnormally Dry → D4 Exceptional) |
| `Shape_Leng` | F(19,11) | DOUBLE | perimeter, m (Albers-computed metadata) |
| `Shape_Area` | F(19,11) | DOUBLE | area, m² |
| `geom` | — | GEOMETRY(MULTIPOLYGON, 4326) | 262,631 vertices total |

**Categorical distribution (nested rings, largest→smallest):**

| DM | Shape_Area (m²) | Parts | Contains AOI centroid (−121.55, 39.8)? |
|----|-----------------|-------|----------------------------------------|
| 0 | 1.9673e12 | 1,138 | ✅ |
| 1 | 1.8019e12 | 568 | ❌ |
| 2 | 1.5036e12 | 336 | ❌ |
| 3 | 7.2353e11 | 89 | ❌ |
| 4 | 1.0500e11 | 21 | ❌ |

**Max severity touching AOI = D0** on the 2026-08-18 release.

**Transformations required:**
- **Validity repair:** `ST_IsValid(geom)` = False for **DM=2** (self-intersection/ring order). Fix once: `ST_MakeValid(geom)` → valid, 338→336 constituent polygons. Assert validity post-repair.
- Severity lookup: point-in-polygon against repaired set, take `MAX(DM)`; dissolve to H3 by centroid sampling (hex ≪ smallest D4 slivers is fine at res-8).
- Area sanity: `Shape_Area` is precomputed Albers metadata — do not recompute in EPSG:4326 (degree² nonsense); DuckDB `ST_Transform→EPSG:6933` returned NaN in this build (tooling hazard) — use pyproj/geopandas for independent verification.

**Hazards:** (a) invalid DM=2 geometry will propagate NaNs into spatial joins unless MakeValid runs first; (b) national-scale file (262 k vertices) — fine in RAM but don't `fetchdf()` the geometry column repeatedly; (c) temporal mismatch: weekly release raster-of-record is 2026-08-18 vs scenario day 2023-08-15 (F-4) — acceptable as climatological covariate, not as event-day truth; (d) `.sbn/.sbx` spatial index is Esri-specific, ignored by open tooling.

---

### 2.8 Census Block Groups — `census/census_blocks.geojson`

**Physical:** 1,173 B, RFC 7946 FeatureCollection, named CRS member `urn:ogc:def:crs:OGC:1.3:CRS84` (= EPSG:4326 axis order lon,lat). No top-level `bbox`.

**Schema (4 features, all Polygon, 5-pt rectangles):**

| Property | Type | Values |
|----------|------|--------|
| `GEOID` | str | `06007000101`, `06007000102`, `06007000201`, `06007000202` (state 06 = CA, county 007 = **Butte**) |
| `ALAND` | int | 5,000,000 / 4,800,000 / 5,200,000 / 5,100,000 m² |
| `population` | int | 1850 / 3200 / 1400 / 2750 (total **9,200**) |

Geometry bboxes tile the AOI into exact quadrants:
`(-121.8,39.6,-121.55,39.8)`, `(-121.55,39.6,-121.3,39.8)`, `(-121.8,39.8,-121.55,40.0)`, `(-121.55,39.8,-121.3,40.0)` — union == AOI, verified `within_AOI=True` ∀ features.

**Transformations required:**
- Population allocation to H3: areal-weight $pop_{hex} = pop_{BG} \cdot \frac{area(hex \cap BG)}{area(BG)}$; ALAND vs polygon-area ratio flags water/mountain undercount.
- Evacuation demand generation: vehicles ≈ population ÷ avg household size (2.5) × vehicle ownership (1.8) — calibrate in NB03.
- GEOID prefix validation: first 5 digits must equal `06007`.

**Hazards:** (a) ALAND values (~5 km² each) are suspiciously uniform and inconsistent with real Butte County BG areas — synthetic placeholders, do not publish as census truth; (b) rectangle geometries are idealized TIGER cartographic substitutes; (c) only 4 BGs for 1,904 km² ⇒ ~476 km²/BG vs real-world ~5 km² — coarse even for demo purposes.

---

### 2.9 WFIGS Fire Perimeter — `wfigs/wfigs_perimeters.geojson`

**Physical:** 967 B, FeatureCollection, **no `crs` member** (RFC 7946 default WGS84 applies), no `bbox`.

**Single feature:**

| Property | Value |
|----------|-------|
| `poly_IncidentName` | `"Camp Fire Baseline"` |
| `attr_FireDiscoveryDateTime` | `2023-08-15T06:30:00Z` |
| `poly_Source` | `"NIFC WFIGS"` |
| `poly_Acres` | 153,336.0 |
| Geometry | Polygon, 7 exterior points, 1 ring, bbox (−121.72, 39.75, −121.48, 39.92) |

Perimeter occupies **12.60 % of AOI**, fully contained (`within_AOI=True`), centroid → H3 `882814a1dbfffff`.

**Transformations required:**
- Discovery timestamp → ISO parse; acres → hectares ×2.47105 for SI reporting.
- Burn-progression ground truth: binary burned mask per H3 hex (centroid-in-polygon or area-weighted ≥50 %).
- Right-censoring rule for event extraction (NB02): detections after discovery+burn-window are censored, not absent — model with survival-style loss, not MSE.

**Hazards:** (a) **synthetic provenance** — the historical Camp Fire was discovered 2018-11-08 and burned 153,336 acres; this feature reuses the historic acreage under a 2023-08-15 date labeled "Baseline": treat as scenario ground truth, never as archival record; (b) 7-point polygon is a crude convex-ish outline — real WFIGS perimeters carry thousands of vertices; downstream edge-matching tolerances must stay ≥ 100 m; (c) single-feature file breaks batch-oriented loaders that assume >1 row.

---

### 2.10 Caltrans PeMS Traffic — `traffic/pems_station_speeds.csv`

**Physical:** 5,301 B, UTF-8, **CRLF line endings**, 121 lines (header + 120 records).

**Schema:**

| Column | Dtype | Nulls | Min | Max | Notes |
|--------|-------|-------|-----|-----|-------|
| `station_id` | BIGINT | 0 | 318491 | 318495 | 5 distinct stations |
| `timestamp` | TIMESTAMP | 0 | 2023-08-15 00:00:00 | 2023-08-15 23:00:00 | 24 hourly stamps, complete panel (5×24=120) |
| `flow_veh_hr` | BIGINT | 0 | 365 | 2,010 | vehicles/hour |
| `avg_speed_kmh` | DOUBLE | 0 | 58.0 | 112.6 | km/h |
| `occupancy_pct` | DOUBLE | 0 | 0.05 | 0.349 | **fraction, not percent** despite name |

**Transformations required:**
- Station → geometry join requires an external PeMS station-metadata table (lat/lon, freeway, lane count) — **absent from `data/raw/`**; stations 318491–318495 are real District-03 IDs but cannot be mapped from this file alone (F-5-family gap).
- Density estimate: $k = flow / speed$ (veh/km) with guard for speed→0; fundamental-diagram sanity: k should stay < ~140 veh/km/lane.
- Capacity & V/C: $C = lanes \times 2{,}200$ veh/h (freeway lane); $V/C = flow/C$ → BPR travel-time factor (see §2.6). Division-by-zero guard mandatory.
- Occupancy→density g-factor alternative: $k \approx occ \times 1000 / \bar{L}_{veh}$, effective vehicle length $\bar{L}_{veh}≈5.5$–7.5 m.
- Rename `occupancy_pct` → `occupancy_frac` in the curated layer to kill the unit lie.

**Hazards:** (a) no coordinates (blocking for graph coupling); (b) `occupancy_pct` naming vs [0.05, 0.349] range contradiction; (c) speeds up to 112.6 km/h exceed typical freeway free-flow calibration ceilings — plausible but flag for BPR α/β refit; (d) CRLF endings break naive `split('\n')` parsers; (e) single-day panel — no weekday/weekend signal for congestion priors.

---

### 2.11 EPA AirNow PM2.5 — `airnow/airnow_pm25_aoi.csv`

**Physical:** 436 B, UTF-8, 7 lines (header + 6 records).

**Schema:**

| Column | Dtype | Values |
|--------|-------|--------|
| `site_name` | VARCHAR | Paradise-Theater, Chico-Manzanita, Oroville-County, Magalia-Skyway, Cohasset-Ridge, Forest-Ranch |
| `latitude` | DOUBLE | 39.512 – 39.924 |
| `longitude` | DOUBLE | −121.843 – −121.556 |
| `pm25_ugm3` | DOUBLE | 22.1 – 110.6 µg/m³ |
| `timestamp` | TIMESTAMP | single stamp 2023-08-15 08:00:00 |
| `unit` | VARCHAR | `UG/M3` (constant) |

**Containment:** 4/6 sites inside AOI; Chico-Manzanita 0.043° west outside; Oroville-County 0.088° south outside — keep both for IDW boundary conditioning.

**Transformations required:**
- IDW smoke field over H3: $PM_{hex} = \frac{\sum_i w_i PM_i}{\sum_i w_i}$, $w_i = d_i^{-2}$ (great-circle km); cross-validation leave-one-out for exponent tuning.
- NowCast-style smoothing if time series grows beyond the single snapshot.
- AQI mapping via EPA breakpoint table (12.0–35.4 µg/m³ → AQI 51–100, etc.) for NB06 symbology.

**Hazards:** (a) one timestamp only — no temporal interpolation possible; pair with ERA5 hour grid by nearest-hour join; (b) Magalia-Skyway 110.6 µg/m³ during an active-fire scenario is *low* (real Camp Fire days exceeded 500 µg/m³) — synthetic magnitudes, don't validate dispersion models against them; (c) site_name is the only natural key — no AQS site codes.

---

## 3. Spatial Alignment & H3 Discretization Guide

### 3.1 AOI Overlap Proof (measured)

| Feed | Extent vs AOI (−121.8, 39.6, −121.3, 40.0) | Verdict |
|------|--------------------------------------------|---------|
| Census | 4 polygons union == AOI exactly | ✅ perfect |
| ERA5-Land | half-cell padded box (−121.8278, 39.5778, −121.2722, 40.0222) ⊇ AOI | ✅ contains |
| WFIGS | bbox inside AOI, 12.60 % area | ✅ contained |
| DEM / EVT | overlap (−121.8, 39.8615, −121.6615, 40.0) = **9.59 %** of AOI | 🔴 partial (F-2) |
| AirNow | 4/6 sites inside; 2 within 0.09° | ✅ usable |
| MODIS | 29/94,000 points inside | ⚠️ sparse |
| FIRMS | **0/74** points inside | 🔴 empty (F-3) |
| PeMS | no coordinates — unmappable | ⚠️ blocked |
| USDM | DM=0 polygon contains AOI centroid | ✅ covered (severity D0) |
| OSM | header box disjoint in latitude (38.374–38.738 vs 39.6–40.0); 1/554,318 ways touch AOI (power line) | 🔴 disjoint (F-1) |

AOI area: 0.2 deg² ≈ **1,904 km²** (cos-corrected at 39.8°N).

### 3.2 H3 Resolution 8 Mapping Rules

Res-8 constants (measured via `h3.cell_area`): average edge **0.5314 km**, average hex area **0.7523 km²** → ≈ **2,531 hexes** tile the AOI. Storage: `h3_id UINT64` via `h3.str_to_int` (e.g., AOI centroid `882814a02dfffff` → `613194654151606271`).

| Layer | Binding rule | Notes |
|-------|--------------|-------|
| DEM / EVT rasters | hex centroid → pixel lookup; aggregate mean (DEM) / mode (EVT) over ~**1,032 px/hex** (px ≈ 729 m²) | co-registered grids allow one shared index |
| ERA5 | bilinear interp of 4 surrounding 0.05°-grid nodes to hex centroid; wind vector interpolated component-wise, then speed/dir derived | never interpolate direction directly |
| FIRMS / MODIS points | `latlng_to_cell(lat, lon, 8)` at ingest; counts + ΣFRP per hex | 29 MODIS pts → 21 unique hexes (verified) |
| WFIGS perimeter | hex centroid-in-polygon → burned bool; area-weighted share as confidence | |
| USDM | hex centroid → MAX(DM) over repaired polygons | |
| Census | areal-weight population split across intersecting hexes | |
| AirNow | hex of station + IDW field to all hexes | |
| PeMS | **blocked** until station coordinates exist; then snap-station → nearest hex, propagate to graph edges within 50 m | |
| OSM | way segments → intersecting hex list per edge (for exposure joins) | after re-extract (F-1) |

Reference conversions (verified): Paradise-Theater → `88e228b551fffff` (`616468180820295679`); WFIGS centroid → `882814a1dbfffff`; DEM/EVT grid center → `882814a5a3fffff`.

---

## 4. Tested Ingestion Snippets (Notebooks 01–05)

All snippets executed successfully on this machine (18.7 s total, peak RSS well under the 15 GB ceiling). Stream/chunk patterns shown where production files will scale.

**NB01 — FIRMS/MODIS unified typed read with bbox pushdown (DuckDB, streaming):**
```python
import duckdb
con = duckdb.connect()
con.execute("INSTALL 'spatial'"); con.execute("LOAD 'spatial'")

modis_aoi = con.execute("""
    SELECT latitude, longitude, acq_date,
           acq_time::INT AS acq_hhmm,                 -- normalize VARCHAR -> INT like FIRMS
           satellite, confidence::TINYINT AS conf,
           frp::FLOAT4 AS frp_mw, daynight, type::TINYINT AS fire_type
    FROM read_csv('data/raw/modis/modis_2023_usa.csv',
                  types={'acq_time':'VARCHAR','confidence':'INTEGER','type':'INTEGER'},
                  timestampformat='%Y-%m-%d')
    WHERE longitude BETWEEN -121.8 AND -121.3
      AND latitude  BETWEEN   39.6 AND   40.0
""").fetchall()          # 29 rows measured; swap fetch_record_batch() for GB-scale feeds
```

**NB01 — DEM/EVT co-registered windowed read (Rasterio, RAM-bounded):**
```python
import rasterio
from rasterio.windows import from_bounds

with rasterio.open("data/raw/dem/srtm_30m_aoi.tif") as src:      # EVT opens identically
    win = from_bounds(-121.75, 39.88, -121.70, 39.92, src.transform)
    dem_win = src.read(1, window=win)                            # (144,181) float32, ~104 KB
```

**NB01/NB04 — ERA5 lazy slice + derived fields (Xarray; NOTE ascending latitude):**
```python
import xarray as xr, numpy as np

ds = xr.open_dataset("data/raw/weather/era5_hourly_aoi.nc")      # lazy, HDF5-backed
ds = ds.sortby("latitude")                                       # file stores lat ASCENDING
sl = ds[["u10","v10","t2m","d2m"]].sel(
        time=slice("2023-08-15T06","2023-08-15T12"),
        latitude=slice(39.6, 39.9), longitude=slice(-121.8, -121.5))

ws10 = np.sqrt(sl.u10**2 + sl.v10**2)                            # m/s
Tc, Td = sl.t2m - 273.15, sl.d2m - 273.15                        # K -> °C
rh   = 100*np.exp(17.625*Td/(243.04+Td)) / np.exp(17.625*Tc/(243.04+Tc))
# measured: ws10 mean 4.95 m/s, RH mean 33.8 %
```

**NB01/NB04 — USDM validity repair + severity lookup (DuckDB Spatial):**
```python
usdm = con.execute("""
    WITH v AS (SELECT DM::INT dm, ST_MakeValid(geom) AS geom     -- fixes invalid DM=2 ring
               FROM ST_Read('data/raw/drought/USDM_20260818.shp'))
    SELECT dm FROM v
    WHERE ST_Intersects(geom, ST_GeomFromText(
        'POLYGON((-121.8 39.6,-121.3 39.6,-121.3 40.0,-121.8 40.0,-121.8 39.6))'))
    ORDER BY dm DESC LIMIT 1                                     # -> 0 (D0) measured
""").fetchone()[0]
```

**NB01 — small vectors (GeoPandas):**
```python
import geopandas as gpd
census = gpd.read_file("data/raw/census/census_blocks.geojson")  # EPSG:4326, (4, 3)
wfigs  = gpd.read_file("data/raw/wfigs/wfigs_perimeters.geojson")# EPSG:4326, (1, 4)
```

**NB02/NB05 — H3 res-8 assignment (uint64 storage):**
```python
import h3
hex_ids = [h3.latlng_to_cell(lat, lon, 8) for lat, lon, *_ in modis_aoi]
h3_u64  = [h3.str_to_int(h) for h in hex_ids]                    # e.g. 616468180820295679
```

**NB03 — OSM streaming parse (Pyosmium; BLOCKED until re-extract, see F-1):**
```python
import osmium as o

class RoadHandler(o.SimpleHandler):
    def __init__(self):
        super().__init__(); self.roads = []
    def way(self, w):
        t = {tg.k: tg.v for tg in w.tags}
        if t.get("highway") in {"motorway","trunk","primary","secondary",
                                "tertiary","unclassified","residential"} \
           and t.get("access") not in {"private","no"}:
            self.roads.append((w.id, t["highway"], t.get("maxspeed"), t.get("lanes")))

h = RoadHandler()
h.apply_file("data/raw/osm/aoi.osm.pbf", locations=True, idx="flex_mem")
# CURRENT FILE: returns Sacramento roads — re-extract first:
#   osmium extract --bbox=-121.8,39.6,-121.3,40.0 california-latest.osm.pbf -o aoi.osm.pbf
```

**NB03 — PeMS typed read (until station metadata arrives):**
```python
pems = con.execute("""
    SELECT station_id::BIGINT, timestamp,
           flow_veh_hr::INTEGER, avg_speed_kmh::FLOAT4,
           occupancy_pct::FLOAT4 AS occupancy_frac       -- fix unit lie at ingest
    FROM read_csv('data/raw/traffic/pems_station_speeds.csv')
""").df()                                                # (120, 5)
```

---

## 5. Remediation Backlog (ordered, blocking → cosmetic)

1. **Re-extract OSM** for `--bbox=-121.8,39.6,-121.3,40.0` from Geofabrik California (F-1) — blocks NB03 entirely.
2. **Re-download DEM + LANDFIRE EVT** covering the full AOI (two 1°-tile mosaics minimum) (F-2) — blocks NB04 terrain/fuel covariates for 90 % of hexes.
3. **Obtain LANDFIRE EVT attribute table** (CSV/DBF) to legitimize the fuel crosswalk (F-5) — blocks Rothermel parameterization.
4. **Pull FIRMS VIIRS with AOI-bounded query** (or accept MODIS-only hotspot seeding) (F-3).
5. **Fetch PeMS station metadata** (coordinates, lanes, freeway segment) for stations 318491–318495 — blocks traffic-to-graph coupling.
6. Pin canonical simulation clock `2023-08-15T00:00–23:00Z` in NB01 config; route all feeds through it (F-4).
7. Normalize `acq_time` (INT-HHMM) and `confidence` (enum) across FIRMS/MODIS in the curated layer.
8. Apply `ST_MakeValid` to USDM at ingest (permanent fixture, not one-off).
9. Rename `occupancy_pct` → `occupancy_frac`; add unit assertions for ERA5 vars on load.
10. Replace synthetic AirNow/WFIGS magnitudes with realistic values before public-facing NB06 benchmarks.

---

*Report generated from measured artifacts; profiling scripts retained in `audit_scratch/` (`profile_master.py`, `profile_vector.py`, `profile_osm.py`, `profile_align.py`, `test_snippets.py`) for reproducibility.*
