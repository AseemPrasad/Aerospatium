# AeroSpatium

**Multi-Modal Wildfire Progression & Evacuation Route Optimization Engine**

Aerospatium predicts time-dependent road impassability during wildfire
events by coupling multi-source environmental and satellite feeds with
physics-informed computational models, then optimizes dynamic,
non-overlapping evacuation corridors on a live road network.

## Features

- **Physics-based fire spread**: Rothermel (1972) / Albini (1976) rate-of-spread
  model, vectorized with Numba on a 4-core CPU
- **Machine learning hazard prediction**: Spatiotemporal GNN (GATv2 + GRU) over
  H3 hexagonal topology, trained with physics-informed loss on sparse satellite
  observations
- **Survival analysis**: XGBoost Cox / AFT models for time-to-burn prediction
  with Harrell's C-index and Brier score evaluation
- **Dynamic evacuation routing**: Time-dependent A* with Haversine heuristics,
  Yen's K-shortest paths for disjoint primary/secondary corridors
- **Learned edge costs**: Monotonic MLP (exponentiated weight constraints) for
  BPR congestion + fire/smoke hazard penalties
- **Real-time API**: FastAPI endpoints for route computation and SSE fire
  propagation streaming

## Hardware Target

| Component | Requirement |
|-----------|-------------|
| CPU | 4+ cores (AMD Ryzen 3 5300U or equivalent) |
| RAM | ≤ 15 GB (CPU-only, no GPU needed) |
| Storage | 60 MB raw data + 50 MB processed |

## Quick Start

```bash
# Install
make install

# Run notebooks in order: V2Notebooks/01 through V2Notebooks/05

# Or start the API server
make run-api
# → http://localhost:8000/docs

# Run tests
make test
```

## Repository Structure

```
aerospatium/
├── src/aerospatium/
│   ├── ca_engine/       # Rothermel physics + CA simulation
│   ├── graph/           # BPR edge costs, OSM builder, A* routing
│   ├── ml/              # Survival models, evaluation, inference
│   ├── spatial/         # DEM processing, H3 indexing, IDW interpolation
│   └── api/             # FastAPI server, Pydantic schemas
├── notebooks/           # Original version A (physics + survival)
├── V2Notebooks/         # Version B (GNN + learned BPR — recommended)
├── data/
│   ├── raw/             # 11 multi-modal data sources
│   └── processed/       # Parquet feature stores & graph pickles
├── models/              # Trained PyTorch / XGBoost models
├── reports/             # Benchmark JSON, interactive HTML maps
├── tests/               # Unit, integration, and performance tests
├── dags/                # Airflow orchestration DAGs
├── configs/             # Base config + data source settings
└── audit_scratch/       # Data profiling & validation scripts
```

## Data Pipeline

```
Raw Sources (11 feeds) → H3 Grid (2,526 hexes) → GNN Hazard Engine
                                              ↓
           BPR Edge-Cost MLP → Dynamic Road Graph → A* Routing
                                              ↓
                      Interactive 3D Dashboard (MapLibre GL)
```

## Operating Modes & Sensor Latency Guarantees

AeroSpatium is designed with a two-tier operational architecture:

1. **Tactical Simulation & Retrospective Benchmark Mode**:
   - Calibrated and validated on historical mega-fire events (e.g. Camp Fire / Butte County).
   - Ingests high-fidelity atmospheric reanalysis (**ERA5-Land**, 0.1° hourly) and polar orbiter passes (**NASA FIRMS MODIS/VIIRS**, 6–12h cadence).
2. **Operational Real-Time Dispatch Mode**:
   - Replaces high-latency reanalysis feeds with operational short-range forecasts (**NOAA HRRR**, 3 km grid, 1-hour updates).
   - Ingests geostationary active fire observations (**GOES-R ABI / GOES-16/18**, 5-minute cadence) mapped directly to the H3 grid topology.

## Key Data Sources

| Source | Format | Role |
|--------|--------|------|
| NASA FIRMS / MODIS | CSV | Active fire detections & burn masks |
| LANDFIRE EVT | GeoTIFF | Surface fuel loading |
| SRTM DEM | GeoTIFF | Terrain slope, aspect, elevation |
| ERA5-Land | NetCDF-4 | Hourly wind, temperature, humidity |
| OpenStreetMap | .osm.pbf | Routable road network |
| USDM | Shapefile | Drought severity index |
| Census TIGER | GeoJSON | Population density for demand |
| WFIGS | GeoJSON | Historical fire perimeters |
| PeMS | CSV | Traffic flow & speed calibration |
| AirNow | CSV | PM2.5 smoke plume concentrations |

## Known Issues

1. **OSM extract is misconfigured** (covers Sacramento, not Butte County) —
   re-run `osmium extract --bbox=-121.8,39.6,-121.3,40.0` before production use
2. See `audit_scratch/data_layer_audit_report_v2.md` for the full data quality report

## License

MIT
