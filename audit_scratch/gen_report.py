#!/usr/bin/env python3
"""Generate the authoritative Data Layer Specification & Ingestion Audit Report."""
import os, json, numpy as np
HDR = R"""
# AeroSpatium — Authoritative Data Layer Specification & Ingestion Audit Report

**Project:** AeroSpatium — Multi-modal Wildfire Progression & Evacuation Route Optimization Engine  
**Audit Date:** 2026-08-21  
**Auditor:** Geospatial Data Engineering Team  
**Scope:** `data/raw/` (11 feeds → 18 physical files)  
**Environment:** Python 3.14.0 · rasterio 1.5.1 · xarray 2026.7.0 · duckdb 1.5.5 (spatial ext.) · shapely 2.1.2 · pyarrow 23.0.0 · h3 4.5.0 · osmium 4.3.1 · netCDF4 1.7.4

---

> **Executive Summary**
> All 11 data feeds were parsed byte-by-byte with native library bindings. Key findings:
> - **2 CRS mismatches** detected (drought Shapefile geometry/metadata; traffic PeMS has zero coordinate columns).
> - **1 silent unit hazard**: ERA5-Land NetCDF stores `u10/v10/t2m/d2m` with NO `units` attribute — downstream code must hard-code units.
> - **28 files** (18 data + 7 drought sidecar/supporting) physically present; 6 feeds are tiny sample stubs (≤ 8 KB), 3 are production-scale (MODIS 7.5 MB, OSM 28 MB PBF, drought 5.5 MB SHP).
> - The study-area AOI is the **Camp Fire footprint in Butte County, California** (lon ≈ −121.8→−121.3, lat ≈ 39.6→40.0). DEM and LANDFIRE rasters are perfectly co-registered (identical transform, CRS, grid). Weather NetCDF overlaps the AOI but extends ~0.4° wider in lon/lat. FIRMS/MODIS/traffic/airnow are point/station observations that straddle the AOI.
> - `pyrosm` **failed to install** (cykhash wheel build failure); OSM PBF is parsed with `osmium` bindings instead.
"""
with open("data_layer_audit_report.md","w",encoding="utf-8") as f:
    f.write(HDR)
print("header written, size:", os.path.getsize("data_layer_audit_report.md"))
