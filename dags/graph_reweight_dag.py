"""Airflow DAG: Dynamic graph reweighting from hazard predictions.

Runs after the GNN hazard engine produces updated burn probabilities.
Recomputes time-stepped BPR edge costs and updates the routable
road graph for downstream evacuation queries.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "aerospatium",
    "depends_on_past": False,
    "start_date": datetime(2026, 8, 15),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

PROCESSED_DIR = os.environ.get("AEROSPATIUM_PROCESSED_DIR", "/app/data/processed")

# Horizon → (column suffix, hours)
HORIZONS = [
    ("t0", 0),
    ("t3", 3),
    ("t6", 6),
    ("t12", 12),
]

TAU_CRITICAL = 0.75
BETA_HAZARD = 3600.0
GAMMA_SMOKE = 100.0


def reweight_graph_edges(**context):
    """Recompute dynamic BPR edge weights from latest hazard predictions."""
    import numpy as np
    import pandas as pd
    from shapely.strtree import STRtree
    import geopandas as gpd

    edges_path = os.path.join(PROCESSED_DIR, "road_edges.parquet")
    hazard_path = os.path.join(PROCESSED_DIR, "h3_hazard_t12.parquet")
    static_path = os.path.join(PROCESSED_DIR, "h3_static_features.parquet")

    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Road edges not found: {edges_path}")
    if not os.path.exists(hazard_path):
        raise FileNotFoundError(f"Hazard predictions not found: {hazard_path}")
    if not os.path.exists(static_path):
        raise FileNotFoundError(f"Static features not found: {static_path}")

    df_edges = pd.read_parquet(edges_path)
    df_hazard = pd.read_parquet(hazard_path)
    df_static = pd.read_parquet(static_path)

    # Build STRtree for spatial join of edges to H3 cells
    gdf_h3 = gpd.GeoDataFrame(
        df_hazard,
        geometry=gpd.GeoSeries.from_wkt(df_static["geometry_wkt"]),
        crs="EPSG:4326",
    )
    strtree = STRtree(gdf_h3.geometry.values)

    # For each edge, find overlapping H3 burn probability at each horizon
    horizon_cols = {}
    for suffix, hours in HORIZONS:
        colname = f"burn_prob_{suffix}" if f"burn_prob_{suffix}" in df_hazard.columns else f"burn_prob_{hours}h"
        horizon_cols[suffix] = colname

    for suffix, hours in HORIZONS:
        col = horizon_cols[suffix]
        if col not in df_hazard.columns:
            continue

        gnn_probs = []
        for _, edge in df_edges.iterrows():
            line_geom = gpd.GeoSeries.from_wkt([edge["geometry_wkt"]]).iloc[0]
            matched_idx = strtree.query(line_geom)

            if len(matched_idx) > 0:
                matched_cells = gdf_h3.iloc[matched_idx]
                if hours == 0:
                    gnn_probs.append(0.0)
                else:
                    gnn_probs.append(float(matched_cells[col].max()))
            else:
                nearest_idx = strtree.nearest(line_geom.interpolate(0.5, normalized=True))
                nearest = gdf_h3.iloc[nearest_idx]
                if hours == 0:
                    gnn_probs.append(0.0)
                else:
                    gnn_probs.append(float(nearest[col]))

        df_edges[f"gnn_burn_prob_{suffix}"] = np.array(gnn_probs, dtype=np.float32)

    # Recompute dynamic weights
    def compute_weight(base_time, burn_prob, pm25):
        if burn_prob >= TAU_CRITICAL:
            return np.float32(1e9)
        fire_delay = 1200.0 * (burn_prob ** 1.5)
        smoke_delay = GAMMA_SMOKE * max(0.0, pm25 - 35.0)
        return np.float32(base_time + fire_delay + smoke_delay)

    for suffix, hours in HORIZONS:
        if f"gnn_burn_prob_{suffix}" not in df_edges.columns:
            continue

        weights = []
        for _, r in df_edges.iterrows():
            bt = r.get("base_time_s", 0.0)
            pm = r.get("pm25_ugm3", 30.0)
            bp = r[f"gnn_burn_prob_{suffix}"]
            weights.append(compute_weight(bt, bp, pm))

        df_edges[f"weight_{suffix}_s"] = weights

    output_path = os.path.join(PROCESSED_DIR, "road_edges.parquet")
    df_edges.to_parquet(output_path, compression="snappy")

    # Count closures per horizon
    for suffix, hours in HORIZONS:
        col = f"weight_{suffix}_s"
        if col in df_edges.columns:
            closures = int((df_edges[col] >= 1e8).sum())
            context["ti"].xcom_push(f"closures_{suffix}", closures)
            print(f"Road closures @ T={hours}h: {closures}/{len(df_edges)}")

    print(f"Graph reweighted: {len(df_edges)} edges → {output_path}")


with DAG(
    dag_id="graph_reweight_hourly",
    default_args=DEFAULT_ARGS,
    description="Dynamic edge-cost recomputation from GNN hazard predictions",
    schedule_interval="0 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["aerospatium", "routing", "graph"],
) as dag:
    reweight_task = PythonOperator(
        task_id="reweight_graph_edges",
        python_callable=reweight_graph_edges,
    )

    reweight_task
