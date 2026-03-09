"""
Fractal Position Embedding
Multi-scale position encoding inspired by fractal geometry.

Standard sinusoidal embeddings use evenly-spaced frequencies; here we use a
fractal (power-law) frequency spectrum controlled by a Hausdorff-like
``fractal_dim`` parameter.  This produces position embeddings with self-similar
multi-scale structure, analogous to the FractalLensJudge research direction.

Architecture
------------
Positions 0 … seq_len-1 are mapped to ``dim_position``-dimensional vectors via
``num_scales`` sinusoidal bands.  The base frequency of band k is

    omega_k = base_freq * scale_factor^(k * fractal_dim)

giving a power-law spacing of scales.  The remaining dimensions are filled with
learned residual offsets that let the model deviate from the pure fractal basis.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class FractalPositionEmbedding(nn.Module):
    """
    Learnable fractal position embedding.

    The embedding combines a fixed (but parameterised) multi-scale sinusoidal
    basis with a small learned residual, allowing the model to both exploit the
    fractal inductive bias and fine-tune positions during training.

    Args:
        max_seq_len: Maximum sequence length.
        dim_position: Output dimensionality of each position vector.
        num_scales: Number of fractal frequency bands.
        fractal_dim: Hausdorff-like dimension controlling frequency growth rate.
            Values > 1 compress higher scales; values < 1 expand them.
        base_freq: Lowest (coarsest) frequency in the spectrum.
        scale_factor: Multiplicative step between adjacent frequency bands.
        learnable_residual: If True, add a small learned residual per position.
    """

    def __init__(
        self,
        max_seq_len: int,
        dim_position: int,
        num_scales: int = 4,
        fractal_dim: float = 1.5,
        base_freq: float = 1.0,
        scale_factor: float = 2.0,
        learnable_residual: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim_position = dim_position
        self.num_scales = num_scales
        self.fractal_dim = fractal_dim

        # ------------------------------------------------------------------ #
        # Pre-compute the fractal sinusoidal basis                            #
        # ------------------------------------------------------------------ #
        # Each scale contributes a sin + cos pair → 2 dims per scale per freq.
        # We distribute dim_position evenly across scales; any remainder goes
        # into the learned residual.
        dims_per_scale = max(2, (dim_position // num_scales) & ~1)  # keep even
        basis_dims = dims_per_scale * num_scales

        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]

        basis_parts = []
        for k in range(num_scales):
            omega = base_freq * (scale_factor ** (k * fractal_dim))
            # Use multiple frequencies within each scale band
            half = dims_per_scale // 2
            div_term = omega / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
            angles = positions * div_term.unsqueeze(0)  # [T, half]
            basis_parts.append(torch.sin(angles))
            basis_parts.append(torch.cos(angles))

        basis = torch.cat(basis_parts, dim=-1)  # [T, basis_dims]

        # Pad or trim to dim_position
        if basis_dims < dim_position:
            pad = torch.zeros(max_seq_len, dim_position - basis_dims)
            basis = torch.cat([basis, pad], dim=-1)
        else:
            basis = basis[:, :dim_position]

        self.register_buffer("basis", basis)  # [max_seq_len, dim_position]

        # ------------------------------------------------------------------ #
        # Learned components                                                  #
        # ------------------------------------------------------------------ #
        # Learnable scale applied to the fractal basis
        self.scale = nn.Parameter(torch.ones(1))

        if learnable_residual:
            self.residual = nn.Parameter(
                torch.zeros(max_seq_len, dim_position) * 0.01
            )
        else:
            self.residual = None

        # Learnable curvature applied on top of the fractal positions
        self.curvature = nn.Parameter(torch.tensor(0.1))

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Return fractal position vectors for a sequence of length ``seq_len``.

        Returns:
            ``[seq_len, dim_position]``
        """
        pos = self.basis[:seq_len] * self.scale  # [seq_len, dim_position]

        if self.residual is not None:
            pos = pos + self.residual[:seq_len]

        # Apply mild curvature modulation (same idea as CurvedPositionEmbedding)
        pos = pos * (1.0 + self.curvature * torch.sin(pos * 0.1))
        return pos
