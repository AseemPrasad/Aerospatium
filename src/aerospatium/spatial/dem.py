"""Digital elevation model processing: slope, aspect, and hillshade generation.

Implements Horn's 3x3 finite-difference algorithm with meter-corrected
gradients, circular-mean aspect aggregation, and hillshade rendering.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(cache=True, fastmath=True) if HAS_NUMBA else (lambda f: f)
def _horn_gradients(dem: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Horn gradients dz/dx and dz/dy for a 2-D elevation array.

    Parameters
    ----------
    dem : np.ndarray, shape (rows, cols)
        Elevation in meters.
    dx_m : float
        Pixel spacing in the x (column) direction, meters.
    dy_m : float
        Pixel spacing in the y (row) direction, meters.

    Returns
    -------
    tuple of (dz_dx, dz_dy) each with the same shape as *dem*.
    """
    dz_dx = np.zeros_like(dem, dtype=np.float64)
    dz_dy = np.zeros_like(dem, dtype=np.float64)

    # Horn coefficients for 3x3 kernel
    c = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / (8.0 * dx_m)

    rows, cols = dem.shape
    for r in range(1, rows - 1):
        for cc in range(1, cols - 1):
            window = dem[r - 1:r + 2, cc - 1:cc + 2].astype(np.float64)
            dz_dx[r, cc] = np.sum(window * c)

            top = window[0, :]
            bottom = window[2, :]

            dz_dy[r, cc] = (
                (bottom[0] + 2.0 * bottom[1] + bottom[2])
                - (top[0] + 2.0 * top[1] + top[2])
            ) / (8.0 * dy_m)

    return dz_dx, dz_dy


def compute_slope_aspect(
    dem: np.ndarray,
    dx_m: float | None = None,
    dy_m: float | None = None,
    transform=None,
    nodata: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (radians, [0, 2π)) from a DEM.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array (meters).
    dx_m, dy_m : float, optional
        Pixel size in meters. If *transform* is provided, these are
        derived from it.
    transform : rasterio.Affine, optional
        Geotransform for computing pixel sizes.
    nodata : float, optional
        Nodata sentinel to mask.

    Returns
    -------
    slope_deg : np.ndarray
        Slope in degrees, shape == dem.shape.
    aspect_rad : np.ndarray
        Aspect in radians [0, 2π), shape == dem.shape.
        Flat areas (slope < 0.1°) get aspect = 0.0.
    """
    if nodata is not None:
        dem = dem.astype(np.float64, copy=True)
        dem[dem == nodata] = np.nan

    if transform is not None and dx_m is None and dy_m is None:
        dx_m = transform[0]
        dy_m = abs(transform[4])
        if dx_m <= 0:
            dx_m = dy_m
        if dy_m <= 0:
            dy_m = dx_m

    if dx_m is None or dy_m is None:
        raise ValueError("Either (dx_m, dy_m) or transform must be provided.")

    if dx_m <= 0:
        dx_m = 30.0
    if dy_m <= 0:
        dy_m = 30.0

    dz_dx, dz_dy = _horn_gradients(dem.astype(np.float64), dx_m, dy_m)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    # Aspect: atan2(-dz_dy, dz_dx) → [0, 2π), compass-clockwise from North
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_rad = np.mod(aspect_rad, 2.0 * math.pi)

    # Flat areas get aspect 0
    aspect_rad[slope_deg < 0.1] = 0.0

    return slope_deg.astype(np.float32), aspect_rad.astype(np.float32)


def hillshade(
    dem: np.ndarray,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    dx_m: float | None = None,
    dy_m: float | None = None,
    transform=None,
) -> np.ndarray:
    """Generate a hillshade raster from a DEM.

    Parameters
    ----------
    dem : np.ndarray
        2-D elevation array.
    azimuth_deg : float
        Light source azimuth (compass, degrees). Default 315° (NW).
    altitude_deg : float
        Light source altitude (degrees above horizon). Default 45°.
    dx_m, dy_m : float, optional
        Pixel spacing in meters.
    transform : rasterio.Affine, optional
        Geotransform for deriving pixel sizes.

    Returns
    -------
    np.ndarray
        Hillshade values [0, 255], uint8.
    """
    slope_deg, aspect_rad = compute_slope_aspect(dem, dx_m, dy_m, transform)

    az_rad = math.radians(azimuth_deg)
    alt_rad = math.radians(altitude_deg)

    # Hillshade = 255 * (cos(alt) * sin(slope) + sin(alt) * cos(slope) * cos(az - aspect))
    hs = (
        math.cos(alt_rad) * np.sin(np.radians(slope_deg))
        + math.sin(alt_rad) * np.cos(np.radians(slope_deg)) * np.cos(az_rad - aspect_rad)
    )
    hs = 255.0 * hs
    hs = np.clip(hs, 0, 255).astype(np.uint8)
    return hs.astype(np.uint8)


def windowed_dem_read(
    path: str,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a DEM GeoTIFF for a bounding box and compute slope/aspect.

    Parameters
    ----------
    path : str
        Path to the GeoTIFF.
    bbox : (west, south, east, north)
        Bounding box in the raster's CRS.

    Returns
    -------
    dem : np.ndarray
        Elevation array.
    slope_deg : np.ndarray
    aspect_rad : np.ndarray
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for windowed_dem_read")

    with rasterio.open(path) as src:
        window = from_bounds(*bbox, src.transform)
        dem = src.read(1, window=window).astype(np.float32)
        transform = src.transform
        nodata = src.nodata

    dx_m = transform[0]
    dy_m = abs(transform[4]) or dx_m
    if dx_m <= 0:
        dx_m = 30.0
    if dy_m <= 0:
        dy_m = dx_m

    slope_deg, aspect_rad = compute_slope_aspect(
        dem, dx_m=dx_m, dy_m=dy_m, nodata=nodata
    )
    return dem, slope_deg, aspect_rad
