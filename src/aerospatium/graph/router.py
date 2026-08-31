"""Time-dependent A* and disjoint evacuation-corridor routing.

Implements Dijkstra/A* on static weights, time-dependent shortest
path with Haversine heuristics, and Yen's K-shortest paths algorithm
for primary + secondary edge-disjoint corridor extraction.
"""
from __future__ import annotations

import heapq
import math

import pandas as pd

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters (haversine formula)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return EARTH_RADIUS_M * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def build_adjacency_from_edges(
    edges_df: pd.DataFrame,
) -> dict[int, list[tuple[int, int, float]]]:
    """Build an adjacency list from an edge DataFrame.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Must contain columns: u, v, edge_id, weight (at desired horizon).

    Returns
    -------
    dict mapping node_id -> list of (neighbor_id, edge_id, weight)
    """
    adj = {}
    for _, row in edges_df.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        eid = int(row["edge_id"])
        w = float(row.get("weight", row.get("base_time_s", 0.0)))
        if u not in adj:
            adj[u] = []
        adj[u].append((v, eid, w))
    return adj


def time_dependent_astar(
    edges_df: pd.DataFrame,
    node_df: pd.DataFrame,
    start: int,
    goal: int,
    departure_time_s: float = 0.0,
    weight_column: str = "base_time_s",
    closed_penalty: float = 1e9,
    heuristic_speed_mps: float = 30.0,
) -> tuple[list[int] | None, list[int] | None, float, float]:
    """Time-dependent A* shortest path search.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Road edges with source/destination nodes and weight columns.
    node_df : pd.DataFrame
        Nodes with at least node_id, lat, lon.
    start, goal : int
        Node IDs for origin and destination.
    departure_time_s : float
        Simulation departure time (used for time-stepped weight selection).
    weight_column : str
        Column name in edges_df to use as traversal weight.
    closed_penalty : float
        Threshold above which edges are treated as closed.
    heuristic_speed_mps : float
        Speed for the Haversine heuristic denominator.

    Returns
    -------
    path_nodes : list[int] or None
    path_edges : list[int] or None
    travel_time : float
    cumulative_risk : float
    """
    node_coords = {}
    for _, row in node_df.iterrows():
        node_coords[int(row["node_id"])] = (row["lat"], row["lon"])

    adj = build_adjacency_from_edges(edges_df.assign(weight=edges_df.get(weight_column, edges_df.get("base_time_s", 0.0))))

    def heuristic(u: int) -> float:
        if u not in node_coords or goal not in node_coords:
            return 0.0
        lat_u, lon_u = node_coords[u]
        lat_g, lon_g = node_coords[goal]
        return haversine_m(lat_u, lon_u, lat_g, lon_g) / heuristic_speed_mps

    pq = []
    f0 = heuristic(start)
    # (f, g, node, path_nodes, path_edges)
    heapq.heappush(pq, (f0, 0.0, start, [start], []))
    visited = {}

    while pq:
        _f, g, u, path_nodes, path_edges = heapq.heappop(pq)

        if u == goal:
            risk = 0.0
            if len(path_edges) > 0:
                matched = edges_df[edges_df["edge_id"].isin(path_edges)]
                risk = float(matched.get("gnn_burn_prob", matched.get("burn_prob_t3", pd.Series([0]))).sum())
            return path_nodes, path_edges, g, risk

        if u in visited and visited[u] <= g:
            continue
        visited[u] = g

        for v, eid, w in adj.get(u, []):
            if w >= closed_penalty:
                continue
            new_g = g + w
            new_f = new_g + heuristic(v)
            new_path_nodes = [*path_nodes, v]
            new_path_edges = [*path_edges, eid]
            heapq.heappush(pq, (new_f, new_g, v, new_path_nodes, new_path_edges))

    return None, None, float("inf"), float("inf")


def time_dependent_astar_multi_horizon(
    edges_df: pd.DataFrame,
    node_df: pd.DataFrame,
    start: int,
    goal: int,
    departure_time_s: float = 0.0,
    horizon_columns: list[str] | None = None,
) -> int:
    """Select the appropriate weight column based on departure time."""
    if horizon_columns is None:
        horizon_columns = ["weight_t0_s", "weight_t3_s", "weight_t6_s", "weight_t12_s"]

    t_hr = departure_time_s / 3600.0
    if t_hr <= 1.5:
        wc = horizon_columns[0]
    elif t_hr <= 4.5:
        wc = horizon_columns[1]
    elif t_hr <= 9.0:
        wc = horizon_columns[2]
    else:
        wc = horizon_columns[-1]

    return time_dependent_astar(
        edges_df, node_df, start, goal,
        departure_time_s=departure_time_s,
        weight_column=wc,
    )


def k_shortest_paths_yen(
    edges_df: pd.DataFrame,
    node_df: pd.DataFrame,
    start: int,
    goal: int,
    k: int = 2,
    weight_column: str = "base_time_s",
) -> list[tuple[list[int], list[int], float, float]]:
    """K-shortest paths via Yen's algorithm with edge-penalty deactivation.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Road edges with weights.
    node_df : pd.DataFrame
        Nodes with coordinates.
    start, goal : int
        Origin and destination node IDs.
    k : int
        Number of paths to find.
    weight_column : str
        Weight column to use.

    Returns
    -------
    list of (path_nodes, path_edges, total_weight, cumulative_risk)
    """
    base_adj = build_adjacency_from_edges(
        edges_df.assign(weight=edges_df.get(weight_column, edges_df.get("base_time_s", 0.0)))
    )

    node_coords = {
        int(row["node_id"]): (row["lat"], row["lon"])
        for _, row in node_df.iterrows()
    }

    paths = []

    def _astar_avoid_edges(nodes_adjs, avoid_edges: set[int], start, goal):
        """A* that avoids certain edge IDs."""
        pq = []
        h0 = _heuristic(start, goal, node_coords)
        heapq.heappush(pq, (h0, 0.0, start, [start], []))
        visited = {}

        while pq:
            _f, g, u, path_nodes, path_edges = heapq.heappop(pq)
            if u == goal:
                return path_nodes, path_edges, g
            if u in visited and visited[u] <= g:
                continue
            visited[u] = g
            for v, eid, w in nodes_adjs.get(u, []):
                if eid in avoid_edges:
                    continue
                if w >= 1e8:
                    continue
                new_g = g + w
                new_f = new_g + _heuristic(v, goal, node_coords)
                heapq.heappush(
                    pq,
                    (new_f, new_g, v, [*path_nodes, v], [*path_edges, eid]),
                )
        return None, None, None

    def _heuristic(u, goal, coords):
        if u in coords and goal in coords:
            lat_u, lon_u = coords[u]
            lat_g, lon_g = coords[goal]
            return haversine_m(lat_u, lon_u, lat_g, lon_g) / 30.0
        return 0.0

    # First (primary) path
    p_nodes, p_edges, p_weight = _astar_avoid_edges(base_adj, set(), start, goal)
    if p_edges is None:
        return []

    paths.append((p_nodes, p_edges, p_weight, 0.0))

    # Subsequent paths with increasing penalties
    for ki in range(1, k):
        penalized_edges = set()
        for prev_path_edges in [p[1] for p in paths]:
            penalized_edges.update(prev_path_edges)

        s_nodes, s_edges, s_weight = _astar_avoid_edges(base_adj, penalized_edges, start, goal)
        if s_edges is None:
            break

        # Compute risk on this path
        s_risk = 0.0
        if s_edges:
            matched = edges_df[edges_df["edge_id"].isin(s_edges)]
            risk_col = "gnn_burn_prob" if "gnn_burn_prob" in matched.columns else "burn_prob_t3"
            if risk_col in matched.columns:
                s_risk = float(matched[risk_col].sum())

        paths.append((s_nodes, s_edges, s_weight, s_risk))

    return paths


def find_disjoint_corridors(
    edges_df: pd.DataFrame,
    node_df: pd.DataFrame,
    start: int,
    goal: int,
    departure_time_s: float = 0.0,
    weight_column: str = "base_time_s",
    max_overlap_frac: float = 0.10,
) -> tuple[dict | None, dict | None]:
    """Find primary and secondary disjoint evacuation corridors.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Road edges with time-stepped weights.
    node_df : pd.DataFrame
        Junction nodes.
    start, goal : int
        Origin and destination.
    departure_time_s : float
    weight_column : str
    max_overlap_frac : float
        Maximum allowed edge overlap between primary and secondary.

    Returns
    -------
    primary_route : dict with keys: nodes, edges, travel_time_s, burn_risk
    secondary_route : dict (or None if no disjoint path exists)
    """
    p_nodes, p_edges, p_time, p_risk = time_dependent_astar(
        edges_df, node_df, start, goal,
        departure_time=departure_time_s,
        weight_column=weight_column,
    )

    if not p_edges:
        return None, None

    # Apply penalty to primary edges and search again
    penalized_edges = edges_df.copy()
    primary_eids = set(p_edges)
    mask = penalized_edges["edge_id"].isin(primary_eids)
    penalized_edges.loc[mask, weight_column] = penalized_edges.loc[mask, weight_column] * 100.0 + 1e6

    s_nodes, s_edges, s_time, s_risk = time_dependent_astar(
        edges_df, node_df, start, goal,
        departure_time=departure_time_s,
        weight_column=weight_column,
    )

    primary = {
        "nodes": p_nodes,
        "edges": p_edges,
        "travel_time_s": p_time,
        "burn_risk": p_risk,
    }

    secondary = None
    if s_edges:
        overlap = len(set(p_edges) & set(s_edges))
        overlap_frac = overlap / max(1, len(p_edges))
        secondary = {
            "nodes": s_nodes,
            "edges": s_edges,
            "travel_time_s": s_time,
            "burn_risk": s_risk,
            "overlap_fraction": overlap_frac,
        }
    else:
        secondary = None

    return primary, secondary
