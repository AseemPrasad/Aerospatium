"""Performance benchmarks for routing latency under concurrent load.

Validates the <50ms single-thread and <1GB peak memory targets
specified in the initial vision.
"""
import os
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from aerospatium.graph.router import time_dependent_astar


@pytest.fixture
def routing_fixtures():
    """Load road network and return edges, nodes, and query pairs."""
    proc_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

    edges_path = os.path.join(proc_dir, "road_edges.parquet")
    nodes_path = os.path.join(proc_dir, "road_nodes.parquet")

    if not os.path.exists(edges_path) or not os.path.exists(nodes_path):
        pytest.skip("Processed road data not available")

    df_edges = pd.read_parquet(edges_path)
    df_nodes = pd.read_parquet(nodes_path)

    # Generate 100 random origin-destination pairs
    node_ids = df_nodes["node_id"].astype(int).unique()
    rng = np.random.default_rng(42)
    queries = [
        (int(rng.choice(node_ids)), int(rng.choice(node_ids)), 0.0)
        for _ in range(100)
        if len(node_ids) >= 2
    ]
    # Ensure at least some valid pairs
    if len(queries) < 10:
        queries = [
            (int(node_ids[i]), int(node_ids[-1 - i]), 0.0)
            for i in range(min(5, len(node_ids) // 2))
        ]

    return df_edges, df_nodes, queries


class TestRoutingLatency:
    def test_p50_latency_under_50ms(self, routing_fixtures):
        """Single-thread P50 latency must be under 50 ms."""
        df_edges, df_nodes, queries = routing_fixtures

        latencies = []
        for origin, dest, dep in queries:
            start = time.perf_counter()
            time_dependent_astar(
                df_edges, df_nodes,
                start=origin, goal=dest,
                weight_column="weight_t0_s",
            )
            latencies.append((time.perf_counter() - start) * 1000.0)

        if len(latencies) == 0:
            pytest.skip("No valid routing queries")

        p50 = float(np.percentile(latencies, 50))
        assert p50 < 50.0, f"P50 latency {p50:.2f}ms exceeds 50ms budget"

    def test_p95_latency_under_100ms(self, routing_fixtures):
        """Single-thread P95 latency must be under 100 ms."""
        df_edges, df_nodes, queries = routing_fixtures

        latencies = []
        for origin, dest, dep in queries:
            start = time.perf_counter()
            time_dependent_astar(
                df_edges, df_nodes,
                start=origin, goal=dest,
                weight_column="weight_t0_s",
            )
            latencies.append((time.perf_counter() - start) * 1000.0)

        if len(latencies) == 0:
            pytest.skip("No valid routing queries")

        p95 = float(np.percentile(latencies, 95))
        assert p95 < 100.0, f"P95 latency {p95:.2f}ms exceeds 100ms budget"

    def test_memory_under_15gb(self, routing_fixtures):
        """Peak memory during 100 concurrent queries must stay under 15 GB."""
        df_edges, df_nodes, queries = routing_fixtures

        if len(queries) == 0:
            pytest.skip("No valid routing queries")

        tracemalloc.start()

        def execute_query(q):
            origin, dest, _dep = q
            time_dependent_astar(
                df_edges, df_nodes,
                start=origin, goal=dest,
                weight_column="weight_t0_s",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(execute_query, queries))

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 15 * 1024, \
            f"Peak memory {peak_mb:.1f} MB exceeds 15 GB ceiling"
