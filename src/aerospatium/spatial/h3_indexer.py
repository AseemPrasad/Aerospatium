"""H3 hexagonal spatial indexing utilities for Aerospatium.

Provides functions for generating H3 res-8 grids over AOIs, converting
between H3 IDs and geometric representations, and aggregating raster
or tabular data onto the hexagonal topology.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import h3
    HAS_H3 = True
except ImportError:  # pragma: no cover
    HAS_H3 = False

try:
    from shapely.geometry import Polygon
    HAS_GEOPANDAS = True
except ImportError:  # pragma: no cover
    HAS_GEOPANDAS = False

_RES = 8


def aoi_hex_grid(
    bbox: tuple[float, float, float, float],
    h3_resolution: int = _RES,
) -> list[str]:
    """Generate all H3 cells whose centroid falls inside the bounding box.

    Parameters
    ----------
    bbox : (min_lon, min_lat, max_lon, max_lat)
    h3_resolution : int
        H3 resolution. Default 8 (~0.75 km² per hex).

    Returns
    -------
    list[str]
        H3 cell IDs in string form.
    """
    if not HAS_H3:
        raise ImportError("h3 library is required for aoi_hex_grid")

    min_lon, min_lat, max_lon, max_lat = bbox
    # Sample a dense grid of centroids and collect unique H3 cells
    n_steps = 500
    lats = np.linspace(min_lat, max_lat, n_steps)
    lons = np.linspace(min_lon, max_lon, n_steps)
    cells = set()
    for lat in lats:
        for lon in lons:
            cell = h3.latlng_to_cell(lat, lon, h3_resolution)
            cells.add(cell)
    return sorted(cells)


def h3_to_polygon(h3_id: str, crs: str = "EPSG:4326") -> Polygon:
    """Convert an H3 cell ID to a Shapely polygon.

    Parameters
    ----------
    h3_id : str
        H3 cell string.
    crs : str
        CRS for the returned geometry metadata.

    Returns
    -------
    shapely.geometry.Polygon
    """
    if not HAS_H3:
        raise ImportError("h3 library is required")
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required")

    boundary = h3.cell_to_boundary(h3_id)
    # cell_to_boundary returns (lat, lon) pairs
    coords = [(lon, lat) for lat, lon in boundary]
    coords.append(coords[0])  # close the polygon
    poly = Polygon(coords)
    poly.crs = crs
    return poly


def h3_centroid_lookup(h3_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Build a lookup of H3 centroid lat/lon.

    Parameters
    ----------
    h3_ids : list[str]
        H3 cell IDs.

    Returns
    -------
    dict mapping h3_id -> (lat, lon)
    """
    if not HAS_H3:
        raise ImportError("h3 library is required")

    return {h_id: h3.cell_to_latlng(h_id) for h_id in h3_ids}


def raster_to_h3_mean(
    raster: np.ndarray,
    transform,
    h3_ids: list[str],
    nodata: float | None = None,
) -> dict[str, float]:
    """Aggregate raster values to H3 cells by mean of intersecting pixels.

    Parameters
    ----------
    raster : np.ndarray, 2-D
        Raster data array.
    transform : rasterio.Affine
        Raster geotransform.
    h3_ids : list[str]
        H3 cell IDs to aggregate into.
    nodata : float, optional
        Nodata sentinel to exclude.

    Returns
    -------
    dict mapping h3_id -> mean raster value.
    """
    if not HAS_H3 or not HAS_GEOPANDAS:
        raise ImportError("h3 and geopandas are required")

    result = {}

    for h_id in h3_ids:
        poly = h3_to_polygon(h_id)
        minx, miny, maxx, maxy = poly.bounds

        # Convert bounds to pixel coordinates
        col_min, row_max = ~transform * (minx, miny)
        col_max, row_min = ~transform * (maxx, maxy)

        row_start = max(0, math.floor(row_min))
        row_end = min(raster.shape[0], math.ceil(row_max) + 1)
        col_start = max(0, math.floor(col_min))
        col_end = min(raster.shape[1], math.ceil(col_max) + 1)

        if row_end <= row_start or col_end <= col_start:
            result[h_id] = np.nan
            continue

        window = raster[row_start:row_end, col_start:col_end]

        if nodata is not None:
            window = window[window != nodata]

        if window.size == 0:
            result[h_id] = np.nan
        else:
            result[h_id] = float(np.nanmean(window))

    return result


def raster_to_h3_mode(
    raster: np.ndarray,
    transform,
    h3_ids: list[str],
    nodata: float | None = None,
) -> dict[str, tuple[int, int]]:
    """Aggregate categorical raster (e.g. fuel model) by mode within each H3 cell.

    Parameters
    ----------
    raster : np.ndarray, 2-D
        Integer/categorical raster.
    transform : rasterio.Affine
    h3_ids : list[str]
    nodata : float, optional

    Returns
    -------
    dict mapping h3_id -> (majority_class, count)
    """
    if not HAS_H3 or not HAS_GEOPANDAS:
        raise ImportError("h3 and geopandas are required")

    from collections import Counter

    result = {}

    for h_id in h3_ids:
        poly = h3_to_polygon(h_id)
        minx, miny, maxx, maxy = poly.bounds

        col_min, row_max = ~transform * (minx, miny)
        col_max, row_min = ~transform * (maxx, maxy)

        row_start = max(0, math.floor(row_min))
        row_end = min(raster.shape[0], math.ceil(row_max) + 1)
        col_start = max(0, math.floor(col_min))
        col_end = min(raster.shape[1], math.ceil(col_max) + 1)

        if row_end <= row_start or col_end <= col_start:
            result[h_id] = (None, 0)
            continue

        window = raster[row_start:row_end, col_start:col_end]

        if nodata is not None:
            window = window[window != nodata]

        if window.size == 0:
            result[h_id] = (None, 0)
        else:
            counter = Counter(window.flat)
            majority_class, count = counter.most_common(1)[0]
            result[h_id] = (int(majority_class), count)

    return result
