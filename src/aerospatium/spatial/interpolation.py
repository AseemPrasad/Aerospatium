"""Spatial interpolation utilities for gridded field estimation.

Implements Inverse Distance Weighting (IDW) interpolation and
Nearest-Neighbour fallback for sparse sensor networks (e.g. AirNow
PM2.5 stations).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


def idw_interpolate(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    power: float = 2.0,
    max_distance: float | None = None,
    fallback_value: float | None = None,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation of scattered point data.

    Parameters
    ----------
    source_points : np.ndarray, shape (M, 2)
        Coordinates of source stations (lat, lon in degrees or projected
        meters, but consistent with target_points).
    source_values : np.ndarray, shape (M,)
        Values at each source station.
    target_points : np.ndarray, shape (K, 2)
        Coordinates of target interpolation points.
    power : float
        IDW power parameter. Higher = more local influence. Default 2.0.
    max_distance : float, optional
        Maximum search radius. Points outside this distance from all
        sources get *fallback_value* (or NaN if fallback_value is None).
    fallback_value : float, optional
        Value to assign to targets with no sources within max_distance.

    Returns
    -------
    np.ndarray, shape (K,)
        Interpolated values at target points.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.ndim == 1:
        source_points = source_points.reshape(-1, 1)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 1)

    M, dim_s = source_points.shape
    K, dim_t = target_points.shape

    if dim_s != dim_t:
        raise ValueError(
            f"Dimension mismatch: sources are {dim_s}D, targets are {dim_t}D"
        )

    if max_distance is not None:
        # Use KD-tree for efficient nearest-neighbour search within radius
        if HAS_SCIPY:
            tree = cKDTree(source_points)
            neighbors = tree.query_ball_point(target_points, r=max_distance)
        else:
            neighbors = [
                np.where(
                    np.linalg.norm(source_points - tp, axis=1) <= max_distance
                )[0]
                for tp in target_points
            ]

        if HAS_SCIPY:
            distances = np.array([
                np.linalg.norm(source_points[idx] - tp, axis=1)
                for idx, tp in zip(neighbors, target_points)
            ], dtype=object)
        else:
            distances = [
                np.linalg.norm(source_points[idx] - tp, axis=1)
                for idx, tp in zip(neighbors, target_points)
            ]
    else:
        neighbors = [np.arange(M) for _ in range(K)]
        distances = [
            np.linalg.norm(source_points - tp, axis=1) for tp in target_points
        ]

    result = np.full(K, np.nan, dtype=np.float64)

    for k in range(K):
        nbrs = neighbors[k]

        if len(nbrs) == 0:
            if fallback_value is not None:
                result[k] = fallback_value
            else:
                result[k] = np.nan
            continue

        dists = distances[k]

        # Handle exact matches
        exact = dists == 0.0
        if np.any(exact):
            result[k] = float(np.mean(source_values[nbrs[exact]]))
            continue

        weights = 1.0 / (dists ** power)
        values = source_values[nbrs]

        weighted_sum = np.sum(weights * values)
        weight_sum = np.sum(weights)

        if weight_sum > 0:
            result[k] = weighted_sum / weight_sum
        else:
            result[k] = fallback_value if fallback_value is not None else np.nan

    return result


def haversine_distance(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Compute great-circle distance (meters) between two points using the
    haversine formula.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like or float
        Latitude and longitude in degrees.

    Returns
    -------
    np.ndarray or float
        Distance in meters.
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    R = 6371000.0
    return R * c


def idw_smoke_plume(
    station_coords: list[tuple[float, float]],
    station_pm25: list[float],
    target_coords: list[tuple[float, float]],
    power: float = 2.0,
    max_distance_km: float = 50.0,
) -> np.ndarray:
    """Specialized IDW interpolation for PM2.5 smoke plume mapping.

    Parameters
    ----------
    station_coords : list of (lat, lon)
        AirNow station coordinates.
    station_pm25 : list of float
        PM2.5 concentration at each station (µg/m³).
    target_coords : list of (lat, lon)
        H3 centroid coordinates to interpolate to.
    power : float
        IDW power. Default 2.0.
    max_distance_km : float
        Maximum interpolation radius in kilometers.

    Returns
    -------
    np.ndarray
        Interpolated PM2.5 values at target coordinates.
    """
    station_pts = np.array(station_coords, dtype=np.float64)
    target_pts = np.array(target_coords, dtype=np.float64)
    values = np.array(station_pm25, dtype=np.float64)

    # Use haversine distance for geographic coordinates
    max_distance_m = max_distance_km * 1000.0

    result = np.full(len(target_pts), np.nan, dtype=np.float64)

    for i, tp in enumerate(target_pts):
        dists = haversine_distance(
            tp[0], tp[1], station_pts[:, 0], station_pts[:, 1]
        )

        # Filter by max distance
        valid = dists <= max_distance_m
        if not np.any(valid):
            result[i] = np.nan
            continue

        valid_dists = dists[valid]
        valid_values = values[valid]

        # Exact match
        if np.any(valid_dists == 0.0):
            result[i] = float(np.mean(valid_values[valid_dists == 0.0]))
            continue

        weights = 1.0 / (valid_dists ** power)
        result[i] = float(np.sum(weights * valid_values) / np.sum(weights))

    return result
