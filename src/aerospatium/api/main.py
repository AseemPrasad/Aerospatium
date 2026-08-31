"""FastAPI application for AeroSpatium evacuation routing.

Provides synchronous route computation and Server-Sent Events (SSE)
for live fire-propagation streaming.
"""
from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from aerospatium.api.schemas import (
    HazardMapResponse,
    RouteRequest,
    RouteResponse,
)
from aerospatium.graph.router import find_disjoint_corridors

PROJECT_ROOT = os.environ.get("AEROSPATIUM_ROOT", os.getcwd())
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

app = FastAPI(
    title="AeroSpatium API",
    description="Wildfire progression & evacuation route optimization engine",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data cache ──────────────────────────────────────────────────────────────
_df_nodes: pd.DataFrame | None = None
_df_edges: pd.DataFrame | None = None
_df_hazard: pd.DataFrame | None = None


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lazy-load and cache static data tables."""
    global _df_nodes, _df_edges, _df_hazard
    if _df_nodes is None:
        _df_nodes = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "road_nodes.parquet"))
    if _df_edges is None:
        _df_edges = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "road_edges.parquet"))
    if _df_hazard is None:
        _df_hazard = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "h3_hazard_t12.parquet"))
    return _df_nodes, _df_edges, _df_hazard


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "version": "0.2.0"}


@app.post("/v1/evacuation/routes", response_model=RouteResponse)
async def compute_evacuation_routes(request: RouteRequest):
    """Compute primary and secondary evacuation corridors.

    Accepts an origin/destination node pair and returns the optimal
    time-dependent A* path plus an edge-disjoint backup corridor.
    """
    df_nodes, df_edges, _df_hazard = _load_data()

    # Validate nodes exist
    if request.origin_node not in df_nodes["node_id"].values:
        raise HTTPException(status_code=404, detail=f"Origin node {request.origin_node} not found")
    if request.destination_node not in df_nodes["node_id"].values:
        raise HTTPException(status_code=404, detail=f"Destination node {request.destination_node} not found")

    primary, secondary = find_disjoint_corridors(
        edges_df=df_edges,
        node_df=df_nodes,
        start=request.origin_node,
        goal=request.destination_node,
        departure_time_s=request.departure_time_s,
    )

    if primary is None:
        raise HTTPException(
            status_code=404,
            detail=f"No route found from {request.origin_node} to {request.destination_node}"
        )

    closures = 0
    for eid in primary.get("edges", []):
        row = df_edges[df_edges["edge_id"] == eid]
        if not row.empty:
            t = request.departure_time_s / 3600.0
            if t <= 1.5:
                w = row.iloc[0].get("weight_t0_s", 0)
            elif t <= 4.5:
                w = row.iloc[0].get("weight_t3_s", 0)
            elif t <= 9.0:
                w = row.iloc[0].get("weight_t6_s", 0)
            else:
                w = row.iloc[0].get("weight_t12_s", 0)
            if w >= 1e8:
                closures += 1

    return RouteResponse(
        primary_corridor=primary,
        secondary_corridor=secondary,
        total_travel_time_s=primary["travel_time_s"],
        total_risk_exposure=primary["burn_risk"],
        road_closures_encountered=closures,
    )


@app.get("/v1/fire/hazard/{horizon_hours}", response_model=HazardMapResponse)
async def get_hazard_map(horizon_hours: int = 12):
    """Return H3 burn-probability raster at a given forecast horizon."""
    df_hazard = _load_data()[2]

    col = f"burn_prob_{horizon_hours}h"
    if col not in df_hazard.columns:
        col = "burn_prob_12h"

    cells = [
        {
            "h3_id": str(row["h3_id"]),
            "burn_prob": float(row[col]),
            "centroid_lon": float(row["centroid_lon"]),
            "centroid_lat": float(row["centroid_lat"]),
        }
        for _, row in df_hazard.iterrows()
    ]

    probs = df_hazard[col].values
    summary = {
        "mean_burn_prob": float(np.mean(probs)),
        "max_burn_prob": float(probs.max()),
        "pct_above_50": float((probs >= 0.5).sum() / len(probs) * 100),
    }

    return HazardMapResponse(
        horizon_hours=float(horizon_hours),
        hazard_cells=cells,
        summary=summary,
    )


@app.get("/v1/fire/propagation/stream")
async def fire_propagation_stream(request: Request, horizon_hours: int = 12):
    """SSE endpoint for live fire-propagation streaming.

    Yields JSON frames at each forecast horizon, suitable for
    real-time dashboard updates.
    """
    df_hazard = _load_data()[2]

    horizons = [0, 3, 6, 12]
    horizons = [h for h in horizons if h <= horizon_hours]

    async def event_stream():
        for h in horizons:
            col = f"burn_prob_{h}h" if f"burn_prob_{h}h" in df_hazard.columns else "burn_prob_0h"
            frame = {
                "timestamp_utc": f"2023-08-15T{h:02d}:00:00Z",
                "horizon_hours": h,
                "total_burned_h3": int((df_hazard[col] >= 0.5).sum()),
            }
            yield f"data: {json.dumps(frame)}\n\n"
            await asyncio.sleep(0.5)

        yield "data: {\"event\": \"complete\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
