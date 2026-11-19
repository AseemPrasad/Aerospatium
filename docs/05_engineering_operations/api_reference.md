# API Reference & Production Runbook

AeroSpatium exposes high-throughput REST and streaming Server-Sent Events (SSE) endpoints via FastAPI.

---

## 1. REST & Streaming API Endpoints

### 1.1 `POST /route`
Computes an optimal time-dependent evacuation route avoiding active hazard zones.

* **Request Body** (`application/json`):
```json
{
  "start_lat": 39.7596,
  "start_lon": -121.6219,
  "goal_lat": 39.7285,
  "goal_lon": -121.8375,
  "departure_time_s": 0.0,
  "horizon_hours": 3
}
```

* **Response Body** (`200 OK`):
```json
{
  "status": "success",
  "travel_time_s": 1420.5,
  "cumulative_burn_risk": 0.042,
  "corridor_length_m": 24150.0,
  "path_nodes": [10142, 10143, 10189, 10204],
  "geometry": {
    "type": "LineString",
    "coordinates": [
      [-121.6219, 39.7596],
      [-121.6450, 39.7510],
      [-121.8375, 39.7285]
    ]
  }
}
```

---

### 1.2 `POST /routes/disjoint`
Computes primary and secondary edge-disjoint evacuation corridors.

* **Request Body**:
```json
{
  "start_lat": 39.7596,
  "start_lon": -121.6219,
  "goal_lat": 39.7285,
  "goal_lon": -121.8375,
  "max_overlap_frac": 0.10
}
```

* **Response Body**:
```json
{
  "primary_route": {
    "travel_time_s": 1420.5,
    "burn_risk": 0.042,
    "path_nodes": [10142, 10143, 10189, 10204]
  },
  "secondary_route": {
    "travel_time_s": 1680.0,
    "burn_risk": 0.011,
    "path_nodes": [10142, 10550, 10582, 10204]
  },
  "edge_overlap_fraction": 0.0
}
```

---

### 1.3 `GET /stream/fire-spread`
Streams real-time wildfire propagation predictions across the H3 grid via Server-Sent Events (SSE).

* **Protocol**: `text/event-stream`
* **Stream Chunk Format**:
```json
event: hazard_update
data: {
  "timestamp": "2026-11-19T22:42:00Z",
  "horizon_h": 1,
  "newly_ignited_hexes": ["8828308281fffff", "8828308285fffff"],
  "severed_road_edges": [4091, 4092],
  "max_flame_length_m": 4.2
}
```

---

## 2. Production Deployment & Verification Runbook

### 2.1 Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/aerospatium/aerospatium.git
cd aerospatium

# 2. Install dependencies (CPU target)
pip install -r requirements.txt
# Or editable developer install:
pip install -e ".[dev]"
```

### 2.2 Running Test Suites
```bash
# Execute unit and integration suites
python -m pytest tests/ -v --tb=short
```

### 2.3 Starting the API Server
```bash
# Start Uvicorn ASGI server with reload on port 8000
python -m uvicorn aerospatium.api.main:app --host 0.0.0.0 --port 8000
# OpenAPI interactive docs available at: http://localhost:8000/docs
```

### 2.4 Continuous Orchestration (Airflow)
Production data pipelines are scheduled via Apache Airflow DAGs located in `dags/`:
* `dags/aerospatium_ingestion_dag.py`: Hourly trigger for weather, active fire detection, and traffic feed pulls.
* `dags/aerospatium_retrain_dag.py`: Daily pipeline updating H3 static stores and recalibrating the monotonic BPR MLP.
