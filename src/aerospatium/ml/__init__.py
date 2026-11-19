"""__init__ for ml package."""
# Lazy / safe imports
from aerospatium.ml.gnn_hazard import (
    NativeEdgeGAT,
    PhysicsInformedSpreadLoss,
    SpatioTemporalGNNSpread,
)

__all__: list[str] = [
    "NativeEdgeGAT",
    "PhysicsInformedSpreadLoss",
    "SpatioTemporalGNNSpread",
]
