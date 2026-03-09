"""
Gravitational Attention Mechanism
Core attention module using physics-inspired gravitational force computation.
Token attention weights are derived from Newton's law of gravitation applied
to learned masses and curved positional coordinates.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GravitationalAttentionHead(nn.Module):
    """
    Single head of gravitational attention.

    Computes attention weights from gravitational forces between tokens:
        F_ij = G * m_i * m_j / (dist(p_i, p_j)^2 + event_horizon)

    with optional curvature applied to the effective inter-token distances and
    Hawking regularization (``max_force``) to prevent runaway collapse.
    """

    def __init__(
        self,
        head_dim: int,
        gravitational_constant: float = 1.0,
        event_horizon: float = 1e-6,
        max_force: Optional[float] = 50.0,
        curvature: float = 0.15,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.event_horizon = event_horizon
        self.max_force = max_force
        self.curvature = curvature

        # Learnable per-head gravitational constant
        self.G = nn.Parameter(torch.tensor(gravitational_constant))

        # Project head slice → scalar mass for each token
        self.mass_proj = nn.Linear(head_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,          # [batch, seq_len, head_dim]
        positions: Optional[torch.Tensor] = None,  # [seq_len, dim_position]
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Compute token masses (strictly positive)
        masses = F.softplus(self.mass_proj(x))  # [batch, seq_len, 1]

        # Pairwise gravitational distances from positions
        if positions is not None:
            diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len, dim_pos]
            dist_sq = (diff ** 2).sum(dim=-1)  # [seq_len, seq_len]

            if self.curvature != 0.0:
                dist_norm = torch.sqrt(dist_sq + self.event_horizon)
                dist_sq = dist_sq * (1.0 + self.curvature * torch.cos(dist_norm))
                dist_sq = dist_sq.clamp(min=self.event_horizon)
            else:
                dist_sq = dist_sq + self.event_horizon

            dist_sq = dist_sq.unsqueeze(0)  # [1, seq_len, seq_len]
        else:
            # Uniform distance = 1 when no positional information is provided
            dist_sq = torch.ones(1, seq_len, seq_len, device=x.device, dtype=x.dtype)

        # F_ij = G * m_i * m_j / dist²
        G = self.G.abs()
        mass_i = masses                          # [batch, seq_len, 1]
        mass_j = masses.transpose(-2, -1)        # [batch, 1, seq_len]
        forces = G * mass_i * mass_j / dist_sq  # [batch, seq_len, seq_len]

        # Hawking regularisation: cap maximum force
        if self.max_force is not None:
            forces = forces.clamp(max=self.max_force)

        attn_weights = F.softmax(forces, dim=-1)  # [batch, seq_len, seq_len]

        # Apply attention to values (token representations)
        out = attn_weights @ x  # [batch, seq_len, head_dim]
        return out, masses.squeeze(-1)  # also return masses for diagnostics


class MultiHeadGravitationalAttention(nn.Module):
    """
    Multi-head gravitational attention.

    Replaces standard QKV projections with physics-inspired gravitational force
    computation.  Each head operates on a ``head_dim``-slice of the model
    dimension and has its own learnable gravitational constant when
    ``different_G_per_head=True``.

    Args:
        dim_model: Total model (embedding) dimension.
        dim_position: Dimensionality of curved/fractal position vectors (informational).
        num_heads: Number of attention heads.
        gravitational_constant: Initial value of G (one per head if different_G_per_head).
        event_horizon: Minimum effective distance² to prevent division-by-zero.
        max_force: Maximum gravitational force (Hawking regularisation). ``None`` disables.
        curvature: Curvature applied to inter-token distances. ``0`` = flat space.
        different_G_per_head: Give each head an independent G parameter.
    """

    def __init__(
        self,
        dim_model: int,
        dim_position: int = 64,
        num_heads: int = 4,
        gravitational_constant: float = 1.0,
        event_horizon: float = 1e-6,
        max_force: Optional[float] = 50.0,
        curvature: float = 0.15,
        different_G_per_head: bool = True,
    ):
        super().__init__()
        if dim_model % num_heads != 0:
            raise ValueError(f"dim_model ({dim_model}) must be divisible by num_heads ({num_heads})")

        self.dim_model = dim_model
        self.dim_position = dim_position
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.event_horizon = event_horizon
        self.max_force = max_force
        self.curvature = curvature

        # Build one attention head per head; initialise G differently per head if requested
        init_Gs = (
            [gravitational_constant * (0.9 ** h) for h in range(num_heads)]
            if different_G_per_head
            else [gravitational_constant] * num_heads
        )
        self.heads = nn.ModuleList([
            GravitationalAttentionHead(
                head_dim=self.head_dim,
                gravitational_constant=g,
                event_horizon=event_horizon,
                max_force=max_force,
                curvature=curvature,
            )
            for g in init_Gs
        ])

        # Output projection to recombine heads
        self.out_proj = nn.Linear(dim_model, dim_model, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[batch, seq_len, dim_model]`` – token representations.
            positions: ``[seq_len, dim_position]`` – curved/fractal positions.

        Returns:
            ``[batch, seq_len, dim_model]`` – attended representations.
        """
        batch, seq_len, _ = x.shape
        head_outputs = []

        for h, head in enumerate(self.heads):
            x_h = x[..., h * self.head_dim:(h + 1) * self.head_dim]  # [batch, seq_len, head_dim]
            out_h, _ = head(x_h, positions)
            head_outputs.append(out_h)

        out = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, dim_model]
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_attention_diagnostics(
        self,
        x: "torch.Tensor | np.ndarray",
        positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return per-head diagnostic statistics (mean force, mean mass, etc.).
        Accepts either a NumPy array or a torch Tensor as input.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        diag: Dict[str, Dict[str, float]] = {}
        with torch.no_grad():
            for h, head in enumerate(self.heads):
                x_h = x[..., h * self.head_dim:(h + 1) * self.head_dim]
                _, masses = head(x_h, positions)
                G_val = head.G.abs().item()
                mean_mass = masses.mean().item()
                diag[f"head_{h}"] = {
                    "mean_mass": mean_mass,
                    "mean_force": G_val * mean_mass ** 2,
                    "G": G_val,
                    "curvature": head.curvature,
                }
        return diag
