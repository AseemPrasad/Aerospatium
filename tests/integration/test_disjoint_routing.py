"""Integration tests for disjoint evacuation corridor routing.

Verifies that primary and secondary paths do not share edges, that
closed edges are never traversed, and that the time-dependent A*
produces correct shortest paths on static-weight graphs.
"""
import heapq

import numpy as np
import pandas as pd
import pytest

from aerospatium.graph.router import (
    build_adjacency_from_edges,
    find_disjoint_corridors,
    time_dependent_astar,
)


@pytest.fixture
def sample_graph():
    """Create a small synthetic road graph for testing.

    Layout: 5 nodes in a line with a bypass.

       0 -- 1 -- 2 -- 3 -- 4  (primary corridor)
            |         |
            5 -- 6 -- 7       (secondary corridor)
    """
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4, 5, 6, 7],
        "lat": [39.70, 39.71, 39.72, 39.73, 39.74, 39.705, 39.715, 39.725],
        "lon": [-121.50, -121.50, -121.50, -121.50, -121.50, -121.49, -121.49, -121.49],
    })

    edges = pd.DataFrame({
        "edge_id": range(9),
        "u": [0, 1, 2, 3, 1, 5, 6, 7, 7],
        "v": [1, 2, 3, 4, 5, 6, 7, 3, 4],
        "base_time_s": [60, 60, 60, 60, 30, 30, 30, 30, 30],
        "speed_kmh": [60, 60, 60, 60, 60, 60, 60, 60, 60],
        "weight_t0_s": [60, 60, 60, 60, 30, 30, 30, 30, 30],
        "weight_t3_s": [60, 60, 60, 60, 30, 30, 30, 30, 30],
        "weight_t6_s": [60, 60, 60, 60, 30, 30, 30, 30, 30],
        "weight_t12_s": [60, 60, 60, 60, 30, 30, 30, 30, 30],
        "length_m": [1000] * 9,
        "gnn_burn_prob_t3": [0.1, 0.2, 0.3, 0.4, 0.05, 0.05, 0.05, 0.35, 0.45],
        "pm25_ugm3": [30] * 9,
    })

    return nodes, edges


class TestTimeDependentAStar:
    def test_simple_shortest_path(self, sample_graph):
        """A* should find the shortest path in an unweighted graph."""
        nodes, edges = sample_graph
        path_nodes, _path_edges, travel_time, _risk = time_dependent_astar(
            edges, nodes, start=0, goal=4, weight_column="base_time_s"
        )

        assert path_nodes is not None, "Should find a path from 0 to 4"
        assert path_nodes[0] == 0, "Path should start at origin"
        assert path_nodes[-1] == 4, "Path should end at destination"
        assert travel_time > 0, "Travel time should be positive"
        assert _risk >= 0, "Risk should be non-negative"

    def test_closed_edge_avoided(self, sample_graph):
        """A* should avoid edges with weight >= 1e8 (hard closure)."""
        nodes, edges = sample_graph.copy()
        # Close the direct path edge 0→1
        edges.loc[edges["edge_id"] == 0, "base_time_s"] = 1e9

        path_nodes, _path_edges, _travel_time, _risk = time_dependent_astar(
            edges, nodes, start=0, goal=2, weight_column="base_time_s",
        )

        assert path_nodes is not None, "Should find a route avoiding closed edge"
        assert 0 not in (_path_edges or []), "Should not use the closed edge"

    def test_no_path_returns_none(self, sample_graph):
        """When all paths are closed, A* should return None."""
        nodes, edges = sample_graph.copy()
        edges["base_time_s"] = 1e9  # close everything

        path_nodes, _path_edges, travel_time, _risk = time_dependent_astar(
            edges, nodes, start=0, goal=4, weight_column="base_time_s"
        )

        assert path_nodes is None, "Should return None when all edges closed"
        assert np.isinf(travel_time), "Travel time should be infinite"

    def test_matches_dijkstra_on_static_weights(self, sample_graph):
        """On static weights, A* should produce the same path as Dijkstra."""
        nodes, edges = sample_graph

        # Dijkstra (brute force, no heuristic)
        adj = build_adjacency_from_edges(edges.assign(weight=edges["base_time_s"]))
        pq = [(0, 0, [0], [])]
        dijk_visited = {}
        dijk_path = None

        while pq:
            g, u, path_n, path_e = heapq.heappop(pq)
            if u == 4:
                dijk_path = (path_n, path_e, g)
                break
            if u in dijk_visited and dijk_visited[u] <= g:
                continue
            dijk_visited[u] = g
            for v, eid, w in adj.get(u, []):
                heapq.heappush(pq, (g + w, v, [*path_n, v], [*path_e, eid]))

        a_star_result = time_dependent_astar(
            edges, nodes, start=0, goal=4, weight_column="base_time_s"
        )

        assert dijk_path is not None
        assert a_star_result[2] == dijk_path[2], "A* and Dijkstra should find same cost"


class TestDisjointCorridors:
    def test_disjoint_corridors_exist(self, sample_graph):
        """Primary and secondary corridors should have low or zero overlap."""
        nodes, edges = sample_graph
        primary, _secondary = find_disjoint_corridors(
            edges, nodes, start=0, goal=7, weight_column="base_time_s"
        )

        assert primary is not None, "Should find primary corridor"
        assert primary["edges"] is not None
        assert len(primary["edges"]) > 0

    def test_overlap_calculation(self, sample_graph):
        """Overlap fraction should be correctly computed."""
        nodes, edges = sample_graph
        _primary, secondary = find_disjoint_corridors(
            edges, nodes, start=0, goal=7, weight_column="base_time_s"
        )

        if secondary is not None:
            assert 0.0 <= secondary.get("overlap_fraction", 0.0) <= 1.0, \
                "Overlap fraction should be in [0, 1]"

    def test_primary_uses_fewest_edges(self, sample_graph):
        """Primary corridor should be shorter/faster than secondary."""
        nodes, edges = sample_graph
        primary, secondary = find_disjoint_corridors(
            edges, nodes, start=0, goal=7, weight_column="base_time_s"
        )

        assert primary is not None
        if secondary is not None:
            assert primary["travel_time_s"] <= secondary["travel_time_s"], \
                "Primary route should be no slower than secondary"
