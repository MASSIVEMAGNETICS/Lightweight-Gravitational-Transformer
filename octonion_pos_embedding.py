"""
Octonion Positional Embeddings
8-dimensional non-associative Octonion embeddings for curved-spacetime
positional encoding in the Lightweight Gravitational Transformer.

Structure of an Octonion: [real, i, j, k, l, il, jl, kl]

By distributing the model dimension across these 8 components with
phase-shifted sinusoids, the embedding captures richer non-Euclidean
geometry than standard sinusoidal positional encodings.  The resulting
``octonion_distance`` function is used in place of the standard Euclidean
distance inside the Polymorphic Attention Orchestrator.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class OctonionEmbedding(nn.Module):
    """
    Generates 8-dimensional Octonion positional embeddings.

    The model dimension is divided evenly across the 8 Octonion components
    (real, i, j, k, l, il, jl, kl).  Each component uses a sinusoid with a
    distinct phase offset ``k * π / 4`` applied on top of the standard
    ``sin(position · div_term)`` basis, approximating the non-Euclidean
    curvature of the Octonion manifold.

    Args:
        dim_model: Total embedding / model dimension.  Should be divisible
            by 8; if not, the last component receives any remaining dims.
        max_len: Maximum sequence length.
    """

    def __init__(self, dim_model: int, max_len: int = 5000):
        super().__init__()
        self.dim_model = dim_model
        self.max_len = max_len

        pe = torch.zeros(max_len, dim_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        # Frequencies at every 8th dimension (one per Octonion component)
        div_term = torch.exp(
            torch.arange(0, dim_model, 8, dtype=torch.float)
            * -(math.log(10000.0) / dim_model)
        )  # [dim_model // 8]

        component_width = dim_model // 8

        for k in range(8):
            start = k * component_width
            # Last component absorbs any remaining dimensions
            end = start + component_width if k < 7 else dim_model
            actual_width = end - start

            # Align div_term to the actual width for this slice
            dt = div_term[:actual_width]
            pe[:, start:end] = torch.sin(position * dt + (k * math.pi / 4))

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return Octonion position embeddings for the input sequence.

        Args:
            x: ``[batch, seq_len, dim_model]`` – token representations
                (used only to determine ``seq_len`` and device).

        Returns:
            ``[1, seq_len, dim_model]`` position embedding tensor on the
            same device as ``x``.
        """
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]  # type: ignore[index]
        if pe.device != x.device:
            pe = pe.to(x.device)
        return pe  # type: ignore[return-value]


def octonion_distance(oct_a: torch.Tensor, oct_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the Octonion norm of the difference between two embedding vectors.

    This replaces the Euclidean ``dist(p_i, p_j)`` in the gravitational force
    formula::

        F_ij = G · m_i · m_j / (dist(p_i, p_j)² + ε)

    The distance is the L2 norm of ``(oct_a - oct_b)`` in the full embedding
    space, with a small ``ε`` (event horizon) added for numerical stability.

    Args:
        oct_a: First set of Octonion vectors ``[..., dim]``.
        oct_b: Second set of Octonion vectors ``[..., dim]`` (broadcast-compatible
            with ``oct_a``).

    Returns:
        Scalar distance tensor ``[..., 1]`` (keepdim).
    """
    diff = oct_a - oct_b
    norm_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)
    return torch.sqrt(norm_sq + 1e-9)


class GravitationalOctonionPosition(nn.Module):
    """
    Computes pairwise Octonion distances between all token positions.

    Wraps :class:`OctonionEmbedding` and :func:`octonion_distance` into a
    single module whose output is a ``[batch, seq_len, seq_len]`` distance
    matrix suitable for use in gravitational force calculations.

    Args:
        dim_model: Model dimension (passed to :class:`OctonionEmbedding`).
        max_len: Maximum sequence length.
    """

    def __init__(self, dim_model: int, max_len: int = 5000):
        super().__init__()
        self.embedding = OctonionEmbedding(dim_model=dim_model, max_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise Octonion distance matrix for all token positions.

        Args:
            x: ``[batch, seq_len, dim_model]`` – token representations.

        Returns:
            ``[batch, seq_len, seq_len]`` non-negative distance matrix.
        """
        # pos: [1, seq_len, dim_model]
        pos = self.embedding(x)

        # Expand for pairwise subtraction
        pos_i = pos.unsqueeze(2)  # [1, seq_len, 1, dim_model]
        pos_j = pos.unsqueeze(1)  # [1, 1, seq_len, dim_model]

        # dist: [1, seq_len, seq_len, 1] → squeeze last dim
        dist = octonion_distance(pos_i, pos_j).squeeze(-1)  # [1, seq_len, seq_len]

        # Broadcast across batch dimension
        batch = x.size(0)
        return dist.expand(batch, -1, -1)  # [batch, seq_len, seq_len]
