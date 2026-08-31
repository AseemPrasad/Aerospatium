"""Pytest configuration and fixtures for the AeroSpatium test suite."""
import os
import sys

import pytest

# Add src to path
PROJECT_ROOT = os.environ.get("AEROSPATIUM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)



@pytest.fixture
def sample_aoi_bbox():
    return (-121.8, 39.6, -121.3, 40.0)


@pytest.fixture
def sample_h3_ids():
    """Generate a small set of H3 cells within the AOI."""
    import h3
    lats = [39.7, 39.8, 39.9, 39.95]
    lons = [-121.7, -121.6, -121.5, -121.45]
    cells = set()
    for lat, lon in zip(lats, lons):
        cells.add(h3.latlng_to_cell(lat, lon, 8))
    return sorted(cells)
