"""
Polymorphic Attention Orchestrator
Implements phase-shifting gravitational attention for the Morphic Cognitive Engine.

The agent transitions between four "cognitive phases" – Solid, Fluid, Gas, and
Singularity – by dynamically reconfiguring the gravitational constant ``G``,
spacetime curvature, and information-density targets.  Force-based attention
weights replace the standard dot-product similarity:

    F_ij = G · m_i · m_j / (dist(p_i, p_j)² + ε)

Distances are computed in 8-dimensional Octonion space via
:class:`~octonion_pos_embedding.GravitationalOctonionPosition`.

Phases
------
- **Solid** (``G=0.5, curvature=0.0``): Precise, low-entropy reasoning.
- **Fluid** (``G=1.0, curvature=0.15``): Balanced general-purpose processing
  (default).
- **Gas** (``G=0.1, curvature=0.8``): Creative / exploratory processing with
  high curvature and diffuse attention.
- **Singularity** (``G=50.0, curvature=-0.1``): Extreme focus on a small
  number of high-mass tokens; stabilised by the Hawking clamp.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from octonion_pos_embedding import GravitationalOctonionPosition


# ---------------------------------------------------------------------------
# Phase configuration map
# ---------------------------------------------------------------------------

PHASE_CONFIG: Dict[str, Dict[str, float]] = {
    "solid":       {"G": 0.5,  "curvature": 0.0,  "hawking_clamp": 50.0},
    "fluid":       {"G": 1.0,  "curvature": 0.15, "hawking_clamp": 50.0},
    "gas":         {"G": 0.1,  "curvature": 0.8,  "hawking_clamp": 50.0},
    "singularity": {"G": 50.0, "curvature": -0.1, "hawking_clamp": 50.0},
}

VALID_PHASES = frozenset(PHASE_CONFIG.keys())


# ---------------------------------------------------------------------------
# Polymorphic Attention Orchestrator
# ---------------------------------------------------------------------------

class PolymorphicAttentionOrchestrator(nn.Module):
    """
    Phase-shifting gravitational attention module.

    Replaces standard dot-product attention with a gravitational force
    calculation.  The active phase is set by calling :meth:`morph` before
    (or during) the forward pass.

    Args:
        dim_model: Model (embedding) dimension.
        num_heads: Number of attention heads.  Must divide ``dim_model``.
        max_len: Maximum sequence length for Octonion position embeddings.
        event_horizon: Small constant ``ε`` to prevent division by zero in
            the force denominator.
        initial_phase: Starting cognitive phase (``"fluid"`` by default).
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int = 4,
        max_len: int = 5000,
        event_horizon: float = 1e-6,
        initial_phase: str = "fluid",
    ):
        super().__init__()

        if dim_model % num_heads != 0:
            raise ValueError(
                f"dim_model ({dim_model}) must be divisible by num_heads ({num_heads})"
            )
        if initial_phase not in VALID_PHASES:
            raise ValueError(
                f"Unknown phase '{initial_phase}'. Valid phases: {sorted(VALID_PHASES)}"
            )

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.event_horizon = event_horizon

        # Current phase parameters (mutable, updated by morph())
        self.current_phase: str = initial_phase
        cfg = PHASE_CONFIG[initial_phase]
        self.G: float = cfg["G"]
        self.curvature: float = cfg["curvature"]
        self.hawking_clamp: float = cfg["hawking_clamp"]

        # Learnable per-head mass projections (one per head)
        self.mass_projs = nn.ModuleList([
            nn.Linear(self.head_dim, 1, bias=False)
            for _ in range(num_heads)
        ])

        # Value projection (one per head, recombined by out_proj)
        self.v_proj = nn.Linear(dim_model, dim_model, bias=False)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=False)

        # Octonion positional distance module
        self.oct_pos = GravitationalOctonionPosition(dim_model=dim_model, max_len=max_len)

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def morph(self, phase: str) -> None:
        """
        Reconfigure attention parameters for the given cognitive phase.

        Args:
            phase: One of ``"solid"``, ``"fluid"``, ``"gas"``,
                ``"singularity"``.

        Raises:
            ValueError: If ``phase`` is not a recognised phase name.
        """
        if phase not in VALID_PHASES:
            raise ValueError(
                f"Unknown phase '{phase}'. Valid phases: {sorted(VALID_PHASES)}"
            )
        cfg = PHASE_CONFIG[phase]
        self.current_phase = phase
        self.G = cfg["G"]
        self.curvature = cfg["curvature"]
        self.hawking_clamp = cfg["hawking_clamp"]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        phase: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute phase-shifted gravitational attention.

        Args:
            x: ``[batch, seq_len, dim_model]`` input representations.
            phase: Optional phase override for this forward pass only.
                Does **not** permanently change :attr:`current_phase`.

        Returns:
            A tuple ``(output, diagnostics)`` where:

            - ``output``: ``[batch, seq_len, dim_model]``
            - ``diagnostics``: dict with keys ``max_force``,
              ``mean_force``, ``phase``, ``G``, ``curvature``.
        """
        if phase is not None:
            if phase not in VALID_PHASES:
                raise ValueError(
                    f"Unknown phase '{phase}'. Valid phases: {sorted(VALID_PHASES)}"
                )
            G = PHASE_CONFIG[phase]["G"]
            curvature = PHASE_CONFIG[phase]["curvature"]
            hawking_clamp = PHASE_CONFIG[phase]["hawking_clamp"]
            active_phase = phase
        else:
            G = self.G
            curvature = self.curvature
            hawking_clamp = self.hawking_clamp
            active_phase = self.current_phase

        batch, seq_len, _ = x.shape

        # Octonion pairwise distance matrix: [batch, seq_len, seq_len]
        dist_matrix = self.oct_pos(x)

        # Apply curvature modulation to distances
        if curvature != 0.0:
            dist_matrix = dist_matrix * (1.0 + curvature * torch.sin(dist_matrix))
            dist_matrix = dist_matrix.clamp(min=0.0)

        # dist² + event_horizon
        dist_sq = dist_matrix ** 2 + self.event_horizon  # [batch, seq_len, seq_len]

        # Per-head gravitational force → aggregate across heads
        v = self.v_proj(x)  # [batch, seq_len, dim_model]
        head_outputs = []
        all_forces = []

        for h, mass_proj in enumerate(self.mass_projs):
            x_h = x[..., h * self.head_dim:(h + 1) * self.head_dim]  # [b, T, hd]

            # Learnable masses (positive via sigmoid)
            masses = torch.sigmoid(mass_proj(x_h))  # [b, T, 1]

            # F_ij = G * m_i * m_j / (dist² + ε)
            mass_i = masses                          # [b, T, 1]
            mass_j = masses.transpose(-2, -1)        # [b, 1, T]
            force = G * (mass_i * mass_j) / dist_sq  # [b, T, T]

            # Hawking clamp (Singularity safety valve)
            force = torch.clamp(force, max=hawking_clamp)
            all_forces.append(force)

            attn = F.softmax(force, dim=-1)  # [b, T, T]

            # Each head attends over the full value projection, then slices
            v_h = v[..., h * self.head_dim:(h + 1) * self.head_dim]  # [b, T, hd]
            head_out = attn @ v_h  # [b, T, hd]
            head_outputs.append(head_out)

        # Recombine heads
        combined = torch.cat(head_outputs, dim=-1)  # [b, T, dim_model]
        output = self.out_proj(combined)

        # Diagnostics
        stacked_forces = torch.stack(all_forces, dim=0)  # [num_heads, b, T, T]
        diagnostics: Dict[str, Any] = {
            "max_force": stacked_forces.max().item(),
            "mean_force": stacked_forces.mean().item(),
            "phase": active_phase,
            "G": G,
            "curvature": curvature,
        }

        return output, diagnostics
