#!/usr/bin/env python3
"""Append section 1: Master Data Catalog & Ingestion Readiness Matrix."""
import os

SECTIONS = []

# Section 1: Master Data Catalog & Ingestion Readiness Matrix
s1 = """## 1. Master Data Catalog & Ingestion Readiness Matrix

Below is the unified catalog for all 11 data feeds (18 physical files including 7 ESRI Shapefile sidecar files in `drought/`). File sizes are exact on-disk byte counts measured via `os.stat`.

| # | Dataset | Target Path | Size (bytes) | MIME / Format | Native CRS | Spatial Extent (BBOX) | Primary Key / Temporal Key | Consumer Notebooks | Ingestion Strategy |
|---|---------|-------------|-------------|---------------|------------|----------------------|---------------------------|--------------------|--------------------|
| 1 | FIRMS VIIRS 375 m | `data/raw/firms/viirs_snpp_california.csv` | 6,275 | CSV / ASCII | WGS84 (lon/lat implicit) | lon [−122.112, −114.254], lat [32.533, 40.822] | `(acq_date, latitude, longitude, frp)` | 01, 02, 05 | DuckDB `read_csv_auto` + spatial filter |
| 2 | MODIS 1 km thermal | `data/raw/modis/modis_2023_usa.csv` | 7,572,283 | CSV / ASCII | WGS84 (lon/lat implicit) | lon [−164.072, −67.380], lat [19.150, 68.202] | `(acq_date, acq_time, latitude, longitude)` | 01, 02, 06 | DuckDB streaming `read_csv_auto`, filter to AOI |
| 3 | LANDFIRE EVT fuel model | `data/raw/landfire/landfire_evt.tif` | 500,762 | GeoTIFF (GTiff) | EPSG:4326 | lon [−121.800, −121.662], lat [39.862, 40.000] | pixel (row, col) → H3-res8 | 01, 02, 03 | Rasterio windowed + masked read |
| 4 | SRTM 30 m DEM | `data/raw/dem/srtm_30m_aoi.tif` | 1,001,134 | GeoTIFF (GTiff) | EPSG:4326 | lon [−121.800, −121.662], lat [39.862, 40.000] | pixel (row, col) → H3-res8 | 01, 02, 03, 04 | Rasterio memory-mapped `open().read(masked=True)` |
| 5 | ERA5-Land hourly | `data/raw/weather/era5_hourly_aoi.nc` | 49,416 | NetCDF-4 / HDF5 | WGS84 (lon/lat) | lon [−121.800, −121.300], lat [39.600, 40.000] | `(time, latitude, longitude)` | 01, 02 | Xarray chunked (dask) disk-slice |
| 6 | OpenStreetMap road net | `data/raw/osm/aoi.osm.pbf` | 28,389,346 | OSM PBF (protobuf) | WGS84 (CRS84) | AOI-extracted (node-bounded) | `osm_id` (node/way/rel) | 01, 04, 05 | `osmium` SimpleHandler parser → networkx |
| 7 | USDM drought | `data/raw/drought/USDM_20260818.shp` (+6 sidecar) | 5,752,736 | ESRI Shapefile | GCS_WGS_1984 (EPSG:4326)¹ | lon [−159.786, −66.539], lat [18.114, 65.108] | `OBJECTID`, `DM` (severity 0–4) | 01, 03, 05 | DuckDB `ST_READ` + spatial clip to AOI |
| 8 | Census TIGER blocks | `data/raw/census/census_blocks.geojson` | 1,173 | GeoJSON (RFC 7946) | CRS84 (OGC URN) | lon [−121.800, −121.300], lat [39.600, 40.000] | `GEOID` (census ID) | 01, 03, 04, 05 | DuckDB `read_json` → `ST_GeomFromWKB` |
| 9 | WFIGS fire perimeters | `data/raw/wfigs/wfigs_perimeters.geojson` | 967 | GeoJSON (RFC 7946) | CRS84 (implicit²) | lon [−121.720, −121.480], lat [39.750, 39.920] | `poly_IncidentName` | 01, 02, 05, 06 | DuckDB `read_json` → `ST_GeomFromWKB` |
| 10 | PeMS traffic speeds | `data/raw/traffic/pems_station_speeds.csv` | 5,301 | CSV / ASCII | WGS84 (lon/lat implicit³) | NO coordinate columns — see §2.10 | `(station_id, timestamp)` | 01, 04, 05 | DuckDB `read_csv_auto` (join ext. station table needed) |
| 11 | AirNow PM2.5 | `data/raw/airnow/airnow_pm25_aoi.csv` | 436 | CSV / ASCII | WGS84 (lon/lat implicit) | lon [−121.843, −121.556], lat [39.512, 39.924] | `(site_name, timestamp)` | 01, 02, 05, 06 | DuckDB `read_csv_auto` |

¹ Drought `.prj` declares `GCS_WGS_1984`, but `Shape_Leng`/`Shape_Area` carry projected-unit magnitudes — CRS/Metadata inconsistency → see §2.7 HAZARD.  
² WFIGS GeoJSON has no `crs` member (defaults to WGS84 per RFC 7946).  
³ PeMS CSV contains **no latitude/longitude columns at all** — station geometry must come from an external PeMS station metadata lookup table → see §2.10 HAZARD.

### Footprint Totals

| Aggregation | Value |
|-------------|-------|
| Total physical files | 18 (11 datasets + 7 drought sidecar) |
| Total on-disk bytes (raw) | 33,678,155 (~32.1 MB) |
| Production-scale feeds (> 1 MB) | MODIS (7.3 MB), OSM PBF (28 MB), Drought SHP (5.5 MB), DEM (1.0 MB) |
| Sample/stub feeds (< 8 KB) | FIRMS, Census, WFIGS, Traffic, AirNow |
"""
SECTIONS.append(s1)
with open("data_layer_audit_report.md", "a", encoding="utf-8") as f:
    f.write("".join(SECTIONS))
print("Section 1 appended. Total lines:", os.path.getsize("data_layer_audit_report.md"))
