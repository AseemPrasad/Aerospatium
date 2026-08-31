"""Tests for the schema validation of API request/response models."""
import pytest
from pydantic import ValidationError

from aerospatium.api.schemas import RouteRequest


class TestRouteRequestSchema:
    def test_valid_request(self):
        req = RouteRequest(origin_node=1001, destination_node=1004, departure_time_s=3600.0)
        assert req.origin_node == 1001
        assert req.destination_node == 1004
        assert req.departure_time_s == 3600.0
        assert req.scenarios == ["baseline"]

    def test_default_departure_time(self):
        req = RouteRequest(origin_node=1, destination_node=2)
        assert req.departure_time_s == 0.0

    def test_negative_departure_rejected(self):
        with pytest.raises(ValidationError):
            RouteRequest(origin_node=1, destination_node=2, departure_time_s=-100)
