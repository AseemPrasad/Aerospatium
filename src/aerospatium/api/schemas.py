"""Pydantic request/response schemas for the AeroSpatium FastAPI endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RouteRequest(BaseModel):
    """Request body for evacuation route computation.

    Example::

        {
            "origin_node": 1001,
            "destination_node": 1004,
            "departure_time_s": 3600.0,
            "scenarios": ["baseline", "high_wind"]
        }
    """

    origin_node: int = Field(..., description="OSM node ID of the evacuation origin")
    destination_node: int = Field(..., description="OSM node ID of the safe-zone destination")
    departure_time_s: float = Field(default=0.0, ge=0.0, description="Departure time in seconds from fire start")
    scenarios: list[str] = Field(default_factory=lambda: ["baseline"])

    @field_validator("departure_time_s")
    @classmethod
    def validate_departure(cls, v):
        if v < 0:
            raise ValueError("Departure time must be non-negative")
        return v


class RouteResponse(BaseModel):
    """Response containing primary and secondary evacuation corridors."""

    primary_corridor: dict[str, Any] = Field(
        ..., description="Primary route: nodes, edges, travel_time_s, burn_risk"
    )
    secondary_corridor: dict[str, Any] | None = Field(
        default=None, description="Secondary backup route with overlap_fraction"
    )
    total_travel_time_s: float = Field(...)
    total_risk_exposure: float = Field(...)
    road_closures_encountered: int = Field(default=0)


class FirePropagationResponse(BaseModel):
    """Response for fire propagation streaming endpoint."""

    timestamp_utc: str = Field(..., description="ISO-8601 timestamp")
    horizon_hours: float = Field(..., ge=0.0, le=12.0)
    h3_hexagons: list[dict[str, Any]] = Field(
        ..., description="List of {h3_id, burn_prob, elevation, fuel}"
    )
    total_burned_h3: int = Field(...)
    area_burned_km2: float = Field(...)


class HazardMapResponse(BaseModel):
    """Response containing a full hazard map for a given horizon."""

    horizon_hours: float
    hazard_cells: list[dict[str, Any]]
    summary: dict[str, float]
