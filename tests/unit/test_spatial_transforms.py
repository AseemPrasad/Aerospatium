"""Unit tests for spatial transform functions: slope, aspect, and H3 conversions."""
import math

import numpy as np

from aerospatium.spatial.dem import compute_slope_aspect, hillshade
from aerospatium.spatial.h3_indexer import (
    aoi_hex_grid,
    h3_centroid_lookup,
    h3_to_polygon,
)
from aerospatium.spatial.interpolation import haversine_distance, idw_interpolate


class TestSlopeAspect:
    """Verify DEM-derived slope and aspect against analytical expectations."""

    def test_flat_plane_zero_slope(self):
        """A perfectly flat plane should have zero slope everywhere."""
        dem = np.ones((5, 5), dtype=np.float32) * 100.0
        slope, _aspect = compute_slope_aspect(dem, dx_m=10.0, dy_m=10.0)
        interior = slope[1:-1, 1:-1]
        assert np.allclose(interior, 0.0, atol=1e-4), "Flat plane should yield zero slope"

    def test_constant_slope(self):
        """A planar incline should produce constant, analytically known slope."""
        # 10m elevation gain over 100m distance
        dx_m, dy_m = 10.0, 10.0
        cols, rows = 11, 11
        dem = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                dem[r, c] = r * (10.0 * dx_m / (rows - 1))  # linear ramp

        slope, _aspect = compute_slope_aspect(dem, dx_m=dx_m, dy_m=dy_m)
        expected_angle = math.degrees(math.atan(10.0 * dx_m / ((rows - 1) * dy_m)))
        # Check interior cells (avoiding edges where Horn is undefined)
        interior_slope = slope[1:-1, 1:-1]
        assert np.allclose(interior_slope, expected_angle, atol=1.0), \
            f"Expected slope ~{expected_angle:.2f}°, got {interior_slope.mean():.2f}°"

    def test_aspect_direction(self):
        """A north-facing slope should have aspect near 0° (north)."""
        dem = np.zeros((5, 5), dtype=np.float32)
        # Elevate the south edge (row 4) — north-facing slope
        dem[4, :] = 10.0

        _slope, aspect = compute_slope_aspect(dem, dx_m=10.0, dy_m=10.0)
        # Interior cells should face north (aspect ≈ 0 or 2π)
        interior_aspect = aspect[1:-1, 1:-1]
        # Aspect 0 = north, 2π also = north (due to mod)
        north_facing = np.sum(
            (interior_aspect < 0.2) | (interior_aspect > (2 * math.pi - 0.2))
        )
        assert north_facing > 0, "South-elevated DEM should have north-facing aspect in interior"

    def test_aspect_wrapped_to_2pi(self):
        """Aspect values should be in [0, 2π)."""
        dem = np.random.rand(10, 10).astype(np.float32) * 100
        _slope, aspect = compute_slope_aspect(dem, dx_m=30.0, dy_m=30.0)
        assert np.all(aspect >= 0), "Aspect should not be negative"
        assert np.all(aspect <= 2 * math.pi), "Aspect should not exceed 2π"

    def test_hillshade_range(self):
        """Hillshade output should be uint8 in [0, 255]."""
        dem = np.random.rand(10, 10).astype(np.float32) * 1000
        hs = hillshade(dem, dx_m=30.0, dy_m=30.0)
        assert hs.dtype == np.uint8, "Hillshade should be uint8"
        assert hs.min() >= 0 and hs.max() <= 255


class TestH3Indexing:
    """Verify H3 grid generation and conversions."""

    def test_aoi_hex_grid_count(self):
        """AOI bounding box should produce ~2,500 hexes at res 8."""
        cells = aoi_hex_grid((-121.8, 39.6, -121.3, 40.0), h3_resolution=8)
        assert 2000 <= len(cells) <= 3000, f"Expected ~2500 cells, got {len(cells)}"

    def test_h3_to_polygon_closed(self):
        """H3 polygon should have closing coordinate."""
        cell = "882814a02dfffff"
        poly = h3_to_polygon(cell)
        coords = list(poly.exterior.coords)
        assert coords[0] == coords[-1], "Polygon should be closed"
        assert len(coords) >= 6, "Hexagon should have at least 6 vertices"

    def test_h3_centroid_roundtrip(self):
        """Cell centroid lookup should return valid lat/lon."""
        cell = "882814a02dfffff"
        lookup = h3_centroid_lookup([cell])
        assert cell in lookup
        lat, lon = lookup[cell]
        assert -90 < lat < 90, f"Invalid latitude: {lat}"
        assert -180 < lon < 180, f"Invalid longitude: {lon}"


class TestInterpolation:
    """Verify haversine and IDW interpolation."""

    def test_haversine_known_distance(self):
        """Haversine distance between two known points."""
        # New York to Los Angeles ~3,935 km
        d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900000 < d < 4000000, f"Expected ~3,935 km, got {d/1000:.0f} km"

    def test_idw_exact_at_source(self):
        """IDW should return the source value at a source point."""
        sources = np.array([[39.7, -121.5], [39.8, -121.4]])
        values = np.array([10.0, 20.0])
        targets = np.array([[39.7, -121.5]])
        result = idw_interpolate(sources, values, targets, power=2.0)
        assert abs(result[0] - 10.0) < 1e-6, "IDW at source should return source value"

    def test_idw_no_sources_returns_nan(self):
        """IDW with no sources should return NaN."""
        result = idw_interpolate(
            np.array([]).reshape(0, 2),
            np.array([]),
            np.array([[39.7, -121.5]]),
            power=2.0,
        )
        assert np.isnan(result[0]), "Empty source should yield NaN"
