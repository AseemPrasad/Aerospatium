# AeroSpatium AGENTS.md

Development environment configuration for the Kilo/AeroSpatium agent.

## Environment

- **Shell**: PowerShell 5.1 (Windows) — use `python` as the Python alias
  (not `python3` or `py`)
- **Python**: 3.14.0 at `C:\Users\aseem\AppData\Local\Programs\Python\Python314\`
- **Project root**: `C:\Users\aseem\OneDrive\Dokumen\aerospatium`
- **Working directory**: Same as project root

## Key Paths

- Source modules: `src/aerospatium/`
- Data: `data/raw/` (11 sources), `data/processed/` (Parquet/NPZ outputs)
- Models: `models/` (`.pt` weights)
- Reports: `reports/` (JSON benchmarks, HTML dashboards)
- V2 notebooks: `V2Notebooks/` (5 notebooks — the active pipeline)
- Original notebooks: `Notebooks/` (6 notebooks)
- Audit scripts: `audit_scratch/`

## Commands

### Run tests
```powershell
python -m pytest tests/ -v --tb=short
```

### Lint & typecheck
```powershell
ruff check src/ tests/
mypy src/aerospatium --ignore-missing-imports
```

### Format code
```powershell
black src/ tests/
```

### Start API server
```powershell
uvicorn aerospatium.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Inspect data files
```powershell
python -c "import pandas as pd; print(pd.read_parquet('data/processed/h3_hazard_t12.parquet').describe())"
python -c "import torch; sd = torch.load('models/spatiotemporal_gnn.pt', map_location='cpu'); print(list(sd.keys()))"
```

## Notebook Execution Order (V2)

1. `V2Notebooks/Notebook 01` — Ingests 11 sources, builds H3 grid + GNN tensors
2. `V2Notebooks/Notebook 02` — Trains GNN hazard engine → `h3_hazard_t12.parquet`
3. `V2Notebooks/Notebook 03` — Trains monotonic BPR MLP → `road_edges.parquet`
4. `V2Notebooks/Notebook 04` — Time-dependent A* + disjoint corridors → `evacuation_routes.json`
5. `V2Notebooks/Notebook 05` — CPU concurrency benchmarks + MapLibre dashboard

## Test Conventions

- Tests use `pytest` framework
- Fixtures defined in `tests/conftest.py`
- Unit tests in `tests/unit/` — physics, spatial, BPR, schemas
- Integration tests in `tests/integration/` — pipeline flow, routing
- Performance tests in `tests/performance/` — latency & memory benchmarks
- Tests that depend on processed data use `@pytest.mark.skipif`

## Code Style

- Line length: 100 characters
- Import style: absolute imports from `aerospatium` package
- Type hints: use `from __future__ import annotations`
- No comments unless the task explicitly requires them
