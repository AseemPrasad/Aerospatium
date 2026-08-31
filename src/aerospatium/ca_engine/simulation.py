"""Cellular automata fire-spread simulation engine.

Uses the Rothermel (1972) / Albini (1976) physics model implemented in
``rothermel.py`` to propagate fire across the H3 hexagonal grid using
an 8-directional neighbor topology with directional wind and slope
alignment.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    import h3
    HAS_H3 = True
except ImportError:  # pragma: no cover
    HAS_H3 = False

# Import the physics core
try:
    from aerospatium.ca_engine.rothermel import rothermel_forward_ros
except ImportError:
    # Fallback: define a stub if the module isn't available
    def rothermel_forward_ros(*args, **kwargs):
        return 0.0


def build_neighbor_graph(
    h3_ids: list[str],
    h3_resolution: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build adjacency lists for 1-ring neighbors of an H3 cell set.

    Parameters
    ----------
    h3_ids : list[str]
        Sorted H3 cell IDs.
    h3_resolution : int
        H3 resolution (used for neighbor lookups).

    Returns
    -------
    edge_src : np.ndarray, shape (E,)
        Source node indices.
    edge_dst : np.ndarray, shape (E,)
        Destination node indices.
    edge_dist_m : np.ndarray, shape (E,)
        Inter-centroid distance in meters.
    """
    if not HAS_H3:
        raise ImportError("h3 library is required")

    id_to_idx = {h_id: i for i, h_id in enumerate(h3_ids)}
    edge_src = []
    edge_dst = []
    edge_dist_m = []

    for i, h_id in enumerate(h3_ids):
        disk = h3.grid_disk(h_id, 1)
        lat_i, lon_i = h3.cell_to_latlng(h_id)

        for nbr in disk:
            if nbr == h_id:
                continue
            if nbr not in id_to_idx:
                continue
            j = id_to_idx[nbr]
            lat_j, lon_j = h3.cell_to_latlng(nbr)

            # Haversine distance
            phi1 = math.radians(lat_i)
            phi2 = math.radians(lat_j)
            dphi = math.radians(lat_j - lat_i)
            dlam = math.radians(lon_j - lon_i)
            a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
            c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
            dist = 6371000.0 * c

            edge_src.append(i)
            edge_dst.append(j)
            edge_dist_m.append(dist)

    return (
        np.array(edge_src, dtype=np.int32),
        np.array(edge_dst, dtype=np.int32),
        np.array(edge_dist_m, dtype=np.float32),
    )


def compute_cell_azimuths(
    h3_ids: list[str],
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> np.ndarray:
    """Compute azimuth (radians, [0, 2π)) from source to destination
    for each edge.

    Parameters
    ----------
    h3_ids : list[str]
    edge_src, edge_dst : np.ndarray

    Returns
    -------
    np.ndarray, shape (E,)
        Azimuth from src to dst in radians.
    """
    if not HAS_H3:
        raise ImportError("h3 library is required")

    lats = np.array([h3.cell_to_latlng(h_id)[0] for h_id in h3_ids])
    lons_arr = np.array([h3.cell_to_latlng(h_id)[1] for h_id in h3_ids])

    dlon = np.radians(lons_arr[edge_dst] - lons_arr[edge_src])

    x = np.sin(dlon) * np.cos(lats[edge_dst])
    y = (
        np.cos(lats[edge_src]) * np.sin(lats[edge_dst])
        - np.sin(lats[edge_src]) * np.cos(lats[edge_dst]) * np.cos(dlon)
    )
    az = np.arctan2(x, y)
    az = np.mod(az, 2.0 * math.pi)
    return az


def run_ca_simulation(
    h3_ids: list[str],
    fuel_models: np.ndarray,
    slopes_rad: np.ndarray,
    aspects_rad: np.ndarray,
    wind_speed_ms: np.ndarray,
    wind_dir_rad: np.ndarray,
    ignition_indices: list[int],
    forecast_horizons_hours: list[int] | None = None,
    time_step_min: float = 5.0,
    max_hours: float = 12.0,
    fuel_moisture: float = 0.08,
) -> np.ndarray:
    """Run the Rothermel cellular automata fire-spread simulation.

    Parameters
    ----------
    h3_ids : list[str]
        H3 cell IDs aligned with all feature arrays.
    fuel_models : np.ndarray, shape (N,)
        Rothermel fuel model number (1-13, 99 for non-burnable).
    slopes_rad : np.ndarray, shape (N,)
        Slope in radians for each cell.
    aspects_rad : np.ndarray, shape (N,)
        Aspect in radians for each cell.
    wind_speed_ms : np.ndarray, shape (N,) or (T,)
        10 m wind speed at each cell or per time step.
    wind_dir_rad : np.ndarray, shape (N,) or (T,)
        10 m wind direction (meteorological FROM-direction) in radians.
    ignition_indices : list[int]
        Indices of ignition cells (set to arrival time 0).
    forecast_horizons_hours : list[int], optional
        Horizons at which to record state. Default [1, 3, 6, 12].
    time_step_min : float
        Simulation time step in minutes.
    max_hours : float
        Maximum simulation duration.
    fuel_moisture : float
        Fuel moisture fraction (0.0-1.0).

    Returns
    -------
    np.ndarray, shape (N,)
        Predicted arrival time in minutes for each H3 cell.
        np.inf for cells that never ignite.
    """
    if forecast_horizons_hours is None:
        forecast_horizons_hours = [1, 3, 6, 12]

    N = len(h3_ids)

    # Build graph topology
    edge_src, edge_dst, edge_dist_m = build_neighbor_graph(h3_ids)

    if len(edge_src) == 0:
        # Fallback: no neighbors, no spread
        arrival = np.full(N, np.inf, dtype=np.float64)
        arrival[ignition_indices] = 0.0
        return arrival

    edge_azimuth = compute_cell_azimuths(h3_ids, edge_src, edge_dst)

    # Wind arrays can be per-cell (fixed across time) or per-time-step
    if wind_speed_ms.ndim == 1 and len(wind_speed_ms) == N:
        ws_per_cell = wind_speed_ms
        wd_per_cell = wind_dir_rad

        def wind_at(t):
            return ws_per_cell, wd_per_cell
    elif wind_speed_ms.ndim == 1:
        raise ValueError("Wind arrays must be per-cell (shape (N,)) or per-timestep (shape (T,))")

    # Midflame wind reduction factor
    def wind_reduction(fuel_model):
        if fuel_model == 99:
            return 0.0
        if fuel_model in (1, 2):
            return 0.4  # grass/timber-grass
        if fuel_model == 4:
            return 0.5  # chaparral
        if fuel_model in (10,):
            return 0.3  # timber litter (sheltered)
        return 0.4

    # Initialize
    current_burned = set(ignition_indices)
    arrival_min = np.full(N, np.inf, dtype=np.float64)
    arrival_min[list(ignition_indices)] = 0.0

    burn_status = np.zeros(N, dtype=np.int8)  # 0=unburned, 1=burning, 2=burned
    burn_status[list(ignition_indices)] = 1

    n_steps = int(max_hours * 60.0 / time_step_min)
    time_step_hr = time_step_min / 60.0

    # Build adjacency: for each source node, list of (dst, distance, azimuth)
    adj_src = {}
    for idx in range(len(edge_src)):
        s = int(edge_src[idx])
        d = int(edge_dst[idx])
        dist = float(edge_dist_m[idx])
        az = float(edge_azimuth[idx])
        if s not in adj_src:
            adj_src[s] = []
        adj_src[s].append((d, dist, az))

    for step in range(n_steps):
        current_time_min = step * time_step_min
        t_step = min(step * time_step_hr, max_hours - time_step_hr)

        ws, wd = wind_at(t_step) if wind_speed_ms.ndim == 1 else (wind_speed_ms[step], wind_dir_rad[step])

        new_ignite_candidates = {}  # dst -> candidate arrival time

        for src_node in sorted(current_burned):
            if src_node not in adj_src:
                continue

            if burn_status[src_node] == 2:
                continue

            fm = int(fuel_models[src_node])
            if fm == 99:
                continue

            w_speed = float(ws[src_node]) if hasattr(ws, '__len__') else float(ws)
            w_dir = float(wd[src_node]) if hasattr(wd, '__len__') else float(wd)
            waf = wind_reduction(fm)

            for dst_node, dist_m, az in adj_src[src_node]:
                if burn_status[dst_node] == 2:
                    continue

                d_fm = int(fuel_models[dst_node])
                if d_fm == 99 or d_fm <= 0:
                    continue

                ros = max(
                    1e-6,
                    rothermel_forward_ros(
                        fuel_model=d_fm,
                        slope_rad=float(slopes_rad[dst_node]),
                        aspect_rad=float(aspects_rad[dst_node]),
                        wind_speed_ms=max(0.0, w_speed * waf),
                        wind_dir_rad=w_dir,
                        neighbor_azimuth_rad=az,
                        fuel_moisture=fuel_moisture,
                    ),
                )

                # ROS in m/min → minutes to cross this edge
                min_to_cross = dist_m / ros

                candidate_arrival = current_time_min + min_to_cross

                if dst_node not in new_ignite_candidates or candidate_arrival < new_ignite_candidates[dst_node]:
                    new_ignite_candidates[dst_node] = candidate_arrival

        # Apply new ignitions
        next_burned = set()
        for dst_node, arrival_time in new_ignite_candidates.items():
            if arrival_time <= current_time_min + time_step_min:
                arrival_min[dst_node] = min(arrival_min[dst_node], arrival_time)
                next_burned.add(dst_node)

        # Update burn status
        for node in current_burned:
            if burn_status[node] == 1:
                # Check if this cell has burned out (simplified)
                burn_status[node] = 2

        for node in next_burned:
            if node not in current_burned:
                burn_status[node] = 1
                arrival_min[node] = min(arrival_min[node], new_ignite_candidates[node])

        current_burned.update(next_burned)

        if not next_burned and len(current_burned) == 0:
            break

        if len(current_burned) >= N * 0.95:
            break

    return arrival_min
