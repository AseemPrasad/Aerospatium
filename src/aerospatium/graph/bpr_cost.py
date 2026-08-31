"""Dynamic edge-cost function for the AeroSpatium evacuation routing graph.

Implements the extended BPR (Bureau of Public Roads) congestion function
with monotonic hazard and smoke penalties:

    W_e(t) = BaseTime_e * (1 + a * (V/C)^g) + b * P_burn + d * SmokePenalty

If burn probability exceeds the critical threshold tau, the edge is
hard-closed (W = 1e9 seconds).
"""
from __future__ import annotations

import importlib.util

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from torch import nn


ALPHA_BPR = 0.15
GAMMA_BPR = 4.0
BETA_HAZARD = 3600.0
DELTA_SMOKE = 100.0


if HAS_TORCH:

    class MonotonicHazardLinear(nn.Linear):
        """Linear layer where weights are forced non-negative via exponentiation.

        Used in the learned edge-cost MLP to guarantee monotonicity of the
        hazard penalty with respect to its inputs.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, torch.exp(self.weight), self.bias)


    class DynamicEdgeCostMLP(nn.Module):
        """Learned monotonic edge-cost model.

        Trained on PeMS traffic data with a monotonic hazard subnet that
        guarantees non-negative fire/smoke penalty weights.
        """

        def __init__(self, road_feat_dim: int = 4, tau_critical: float = 0.75):
            super().__init__()
            self.tau_critical = tau_critical
            self.device = torch.device("cpu")

            self.traffic_subnet = nn.Sequential(
                nn.Linear(road_feat_dim, 16),
                nn.SiLU(),
                nn.Linear(16, 8),
                nn.SiLU(),
                nn.Linear(8, 1),
                nn.Softplus(),
            )

            self.hazard_subnet = nn.Sequential(
                MonotonicHazardLinear(2, 4, bias=False),
                nn.ReLU(),
                MonotonicHazardLinear(4, 1, bias=False),
            )

        def forward(
            self,
            base_time: torch.Tensor,
            road_features: torch.Tensor,
            burn_prob: torch.Tensor,
            pm25_val: torch.Tensor,
        ) -> torch.Tensor:
            """Compute dynamic edge weights.

            Parameters
            ----------
            base_time : [E]
                Free-flow travel time in seconds.
            road_features : [E, 4]
                (lanes/4, v/c_ratio, occupancy_frac, speed_norm)
            burn_prob : [E]
                Fire burn probability [0, 1].
            pm25_val : [E]
                PM2.5 concentration (micrograms per m3).

            Returns
            -------
            torch.Tensor, [E]
                Edge weight in seconds.
            """
            traffic_delay = self.traffic_subnet(road_features).squeeze(-1)
            smoke_excess = torch.relu(pm25_val - 35.0) / 100.0
            hazard_in = torch.stack([burn_prob, smoke_excess], dim=-1)
            hazard_delay = self.hazard_subnet(hazard_in).squeeze(-1)
            w_e = base_time + traffic_delay + hazard_delay

            closed_mask = burn_prob >= self.tau_critical
            w_e = torch.where(
                closed_mask,
                torch.tensor(1e9, device=w_e.device),
                w_e,
            )
            return w_e

        def predict_numpy(
            self,
            base_time: np.ndarray,
            road_features: np.ndarray,
            burn_prob: np.ndarray,
            pm25_val: np.ndarray,
        ) -> np.ndarray:
            """NumPy-friendly inference wrapper."""
            with torch.no_grad():
                bt = torch.tensor(base_time, dtype=torch.float32)
                rf = torch.tensor(road_features, dtype=torch.float32)
                bp = torch.tensor(burn_prob, dtype=torch.float32)
                pm = torch.tensor(pm25_val, dtype=torch.float32)
                return self.forward(bt, rf, bp, pm).numpy()

else:  # pragma: no cover

    class MonotonicHazardLinear:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for MonotonicHazardLinear")


    class DynamicEdgeCostMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DynamicEdgeCostMLP")


def bpr_cost(
    base_time_s: float,
    volume: float,
    capacity: float,
    burn_prob: float,
    pm25_ugm3: float,
    alpha: float = ALPHA_BPR,
    gamma: float = GAMMA_BPR,
    beta: float = BETA_HAZARD,
    delta: float = DELTA_SMOKE,
    tau_critical: float = 0.75,
) -> float:
    """Hand-tuned extended BPR edge cost (scalar version).

    Parameters
    ----------
    base_time_s : float
        Free-flow travel time in seconds.
    volume : float
        Current vehicle flow (veh/hr).
    capacity : float
        Edge capacity (veh/hr).
    burn_prob : float
        Burn probability [0, 1].
    pm25_ugm3 : float
        PM2.5 concentration (micrograms/m3).
    alpha, gamma, beta, delta : float
        BPR and penalty coefficients.
    tau_critical : float
        Critical burn probability threshold for road closure.

    Returns
    -------
    float
        Edge traversal weight in seconds.
    """
    if capacity > 0 and volume > 0:
        vc = volume / capacity
        congestion_factor = 1.0 + alpha * (vc ** gamma)
    else:
        congestion_factor = 1.0

    hazard_penalty = beta * burn_prob
    smoke_penalty = delta * max(0.0, (pm25_ugm3 - 35.0) / 100.0)

    w = base_time_s * congestion_factor + hazard_penalty + smoke_penalty

    if burn_prob >= tau_critical:
        w = 1e9

    return w


def bpr_cost_vectorized(
    base_time_s: np.ndarray,
    volume: np.ndarray,
    capacity: np.ndarray,
    burn_prob: np.ndarray,
    pm25_ugm3: np.ndarray,
    alpha: float = ALPHA_BPR,
    gamma: float = GAMMA_BPR,
    beta: float = BETA_HAZARD,
    delta: float = DELTA_SMOKE,
    tau_critical: float = 0.75,
) -> np.ndarray:
    """Vectorized version of :func:`bpr_cost` for batch edge weights.

    Parameters
    ----------
    All arrays must be broadcastable.

    Returns
    -------
    np.ndarray
        Edge weights in seconds.
    """
    base_time_s = np.asarray(base_time_s, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    burn_prob = np.asarray(burn_prob, dtype=np.float64)
    pm25_ugm3 = np.asarray(pm25_ugm3, dtype=np.float64)

    vc = np.divide(
        volume,
        capacity,
        out=np.zeros_like(volume, dtype=np.float64),
        where=capacity > 0,
    )
    congestion_factor = 1.0 + alpha * (vc ** gamma)

    hazard_penalty = beta * burn_prob
    smoke_penalty = delta * np.maximum(0.0, (pm25_ugm3 - 35.0) / 100.0)

    w = base_time_s * congestion_factor + hazard_penalty + smoke_penalty

    closed_mask = burn_prob >= tau_critical
    w = np.where(closed_mask, 1e9, w)

    return w
