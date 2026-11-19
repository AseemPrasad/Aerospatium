"""Spatiotemporal Graph Neural Network for Wildfire Hazard Prediction.

Implements edge-conditioned spatiotemporal graph attention (GATv2 / NativeEdgeGAT + GRU)
trained with a Physics-Informed loss function incorporating Rothermel flux constraints
and irreversible state monotonicity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class NativeEdgeGAT(nn.Module):
    """Native PyTorch implementation of Edge-Conditioned Graph Attention."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 3):
        super().__init__()
        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        h_src = self.lin_src(x[src])
        h_dst = self.lin_dst(x[dst])
        h_e = self.lin_edge(edge_attr)

        alpha = (h_src + h_dst + h_e) * self.att
        alpha = self.leaky_relu(alpha.sum(dim=-1, keepdim=True))

        alpha = torch.exp(alpha - alpha.max())
        alpha_sum = torch.zeros(x.size(0), 1, device=x.device).index_add_(0, dst, alpha) + 1e-6
        alpha_norm = alpha / alpha_sum[dst]

        msg = (h_src + h_e) * alpha_norm
        out = torch.zeros(x.size(0), h_src.size(-1), device=x.device).index_add_(0, dst, msg)
        return out


class SpatioTemporalGNNSpread(nn.Module):
    """SpatioTemporal Graph Neural Network for wildfire spread prediction over H3 grids."""

    def __init__(self, in_node_dim: int = 15, edge_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        if HAS_PYG:
            self.gat1 = GATv2Conv(in_node_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=False)
            self.gat2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=2, concat=False)
        else:
            self.gat1 = NativeEdgeGAT(in_node_dim, hidden_dim, edge_dim=edge_dim)
            self.gat2 = NativeEdgeGAT(hidden_dim, hidden_dim, edge_dim=edge_dim)

        self.recurrent_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(p=0.1)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x_static: torch.Tensor,
        x_dyn_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass predicting burn probabilities across time horizons.

        Parameters
        ----------
        x_static : torch.Tensor, shape [N, static_dim]
        x_dyn_seq : torch.Tensor, shape [T, N, dyn_dim]
        edge_index : torch.Tensor, shape [2, E]
        edge_attr : torch.Tensor, shape [E, edge_dim]

        Returns
        -------
        torch.Tensor, shape [T, N] (burn probability per cell per timestep)
        """
        T, N, _ = x_dyn_seq.shape
        h_t = torch.zeros(N, self.hidden_dim, device=x_static.device)
        burn_probs = []

        for t in range(T):
            x_t = torch.cat([x_static, x_dyn_seq[t]], dim=-1)
            g_t = self.dropout(torch.relu(self.gat1(x_t, edge_index, edge_attr)))
            g_t = self.dropout(torch.relu(self.gat2(g_t, edge_index, edge_attr)))

            h_t = self.recurrent_cell(g_t, h_t)
            p_t = self.hazard_head(h_t).squeeze(-1)
            burn_probs.append(p_t)

        return torch.stack(burn_probs, dim=0)


class PhysicsInformedSpreadLoss(nn.Module):
    """Physics-informed loss enforcing supervised BCE, physical diffusion flux, and temporal monotonicity."""

    def __init__(self, lambda_phys: float = 0.3, lambda_mono: float = 0.8):
        super().__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.lambda_phys = lambda_phys
        self.lambda_mono = lambda_mono

    def forward(
        self,
        p_pred: torch.Tensor,
        y_obs_mask: torch.Tensor,
        y_obs_val: torch.Tensor,
        rothermel_ros: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unreduced_bce = self.bce(p_pred, y_obs_val)
        l_sup = (unreduced_bce * y_obs_mask).sum() / (y_obs_mask.sum() + 1e-6)

        src, dst = edge_index
        spatial_flux = torch.abs(p_pred[:-1, src] - p_pred[:-1, dst])
        l_phys = torch.mean(torch.relu(spatial_flux - rothermel_ros[:-1, src]))

        l_mono = torch.mean(torch.relu(p_pred[:-1, :] - p_pred[1:, :]))
        total_loss = l_sup + (self.lambda_phys * l_phys) + (self.lambda_mono * l_mono)
        return total_loss, l_sup, l_phys, l_mono
