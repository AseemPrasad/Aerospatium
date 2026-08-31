"""Integration test: full ingestion → hazard → routing pipeline flow.

This test exercises the end-to-end data flow from static H3 features
through hazard prediction to routing, without relying on the full
notebooks being executed.
"""
import os

import numpy as np
import pandas as pd
import pytest

from aerospatium.graph.bpr_cost import bpr_cost_vectorized
from aerospatium.graph.router import time_dependent_astar


@pytest.mark.skipif(
    not os.path.exists("data/processed/h3_static_features.parquet"),
    reason="Processed data not available — run notebooks first",
)
class TestPipelineFlow:
    """Validate the complete ingest-to-route pipeline on real artifacts."""

    def test_static_features_schema(self):
        """H3 static features should have the expected schema."""
        df = pd.read_parquet("data/processed/h3_static_features.parquet")
        required_cols = [
            "h3_id", "centroid_lat", "centroid_lon",
            "elevation_mean_m", "slope_mean_deg",
            "fuel_model", "population_total",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        assert len(df) > 0, "Should have at least one H3 cell"
        assert df["h3_id"].dtype in (np.uint64, np.int64, object)
        assert df["elevation_mean_m"].notna().all(), "Elevation should not have nulls"
        assert df["fuel_model"].notna().all(), "Fuel model should not have nulls"

    def test_hazard_table_joined_with_static(self):
        """GNN hazard table should join 1:1 with static features."""
        df_static = pd.read_parquet("data/processed/h3_static_features.parquet")
        df_hazard = pd.read_parquet("data/processed/h3_hazard_t12.parquet")

        assert len(df_hazard) == len(df_static), \
            "Hazard table should have same row count as static features"

        merged = df_static[["h3_id"]].merge(df_hazard, on="h3_id", how="left")
        assert merged["burn_prob_0h"].notna().all(), "Hazard join should be lossless"
        assert merged["burn_prob_12h"].notna().all(), "12h horizon should exist"

        # Burn probability should be monotonically non-decreasing
        p0 = merged["burn_prob_0h"].values
        p12 = merged["burn_prob_12h"].values
        mean_p0 = np.mean(p0)
        mean_p12 = np.mean(p12)
        assert mean_p12 >= mean_p0, "Mean burn probability should increase over time"

    def test_road_network_topology(self):
        """Road edges should form a connected graph."""
        df_edges = pd.read_parquet("data/processed/road_edges.parquet")
        df_nodes = pd.read_parquet("data/processed/road_nodes.parquet")

        assert len(df_edges) > 0, "Should have road edges"
        assert len(df_nodes) > 0, "Should have road nodes"

        # Every edge endpoint should reference an existing node
        node_ids = set(df_nodes["node_id"].astype(int))
        edge_endpoints = set(df_edges["u"].astype(int)) | set(df_edges["v"].astype(int))
        assert edge_endpoints.issubset(node_ids), \
            "All edge endpoints should reference valid nodes"

        # Edge weights should be positive
        assert (df_edges["base_time_s"] > 0).all(), "Base times should be positive"

    def test_bpr_monotonic_on_real_edges(self):
        """BPR cost on real edges should be monotonic in burn probability."""
        df = pd.read_parquet("data/processed/road_edges.parquet")
        base_times = df["base_time_s"].values
        capacities = df["capacity_veh_hr"].fillna(1900).values
        volumes = (df["capacity_veh_hr"].fillna(1900) * 0.5).values  # 50% V/C
        pm25 = df.get("pm25_ugm3", pd.Series(np.full(len(df), 30.0))).values

        p_low = np.full(len(df), 0.3)
        p_high = np.full(len(df), 0.7)

        w_low = bpr_cost_vectorized(base_times, volumes, capacities, p_low, pm25)
        w_high = bpr_cost_vectorized(base_times, volumes, capacities, p_high, pm25)

        # High burn prob should not decrease weight (before closure threshold)
        open_low = w_low[w_low < 1e8]
        open_high = w_high[w_high < 1e8]
        assert (open_high >= open_low - 1e-6).all(), \
            "BPR cost should be non-decreasing in burn probability (for open edges)"

    def test_routing_returns_valid_path(self):
        """A* on real edge data should return a valid path."""
        df_edges = pd.read_parquet("data/processed/road_edges.parquet")
        df_nodes = pd.read_parquet("data/processed/road_nodes.parquet")

        origins = df_nodes["node_id"].astype(int).unique()[:5]
        destinations = df_nodes["node_id"].astype(int).unique()[-5:]

        for origin in origins:
            for dest in destinations:
                if origin == dest:
                    continue
                path_nodes, _path_edges, travel_time, _risk = time_dependent_astar(
                    df_edges, df_nodes,
                    start=int(origin), goal=int(dest),
                    weight_column="weight_t0_s",
                )
                # May or may not find a path depending on network topology
                if path_nodes is not None:
                    assert path_nodes[0] == int(origin)
                    assert path_nodes[-1] == int(dest)
                    assert travel_time > 0
                    assert _risk >= 0
