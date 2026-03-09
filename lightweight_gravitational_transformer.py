"""
Lightweight Gravitational Transformer (LGT)
A minimal, physics-aware transformer using Black Hole Framework attention.
Designed for VictorOS integration, low-compute inference, and recursive meta-cognition.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple

from gravitational_attention import MultiHeadGravitationalAttention
from fractal_position_embedding import FractalPositionEmbedding


# ---------------------------------------------------------------------------
# Curved position embedding (standard, non-fractal variant)
# ---------------------------------------------------------------------------

class CurvedPositionEmbedding(nn.Module):
    """
    Learnable positions on a curved manifold.
    Replaces standard sinusoidal/learnable positional encodings.
    """

    def __init__(self, max_seq_len: int, dim_position: int, curvature: float = 0.15):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(max_seq_len, dim_position) * 0.02)
        self.curvature_scale = nn.Parameter(torch.tensor(curvature))

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return curved position vectors of shape ``[seq_len, dim_position]``."""
        positions = self.positions[:seq_len]
        curved = positions * (1.0 + self.curvature_scale * torch.sin(positions * 0.1))
        return curved


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------

class LightweightGravitationalBlock(nn.Module):
    """
    Single transformer block with gravitational attention + minimal FFN.
    """

    def __init__(
        self,
        dim_model: int = 128,
        dim_position: int = 64,
        num_heads: int = 4,
        ff_expansion: float = 2.0,
        gravitational_constant: float = 1.0,
        curvature: float = 0.15,
        event_horizon: float = 1e-6,
        max_force: Optional[float] = 50.0,
        dropout: float = 0.1,
        learnable_masses: bool = True,
    ):
        super().__init__()

        # Gravitational attention (drop-in replacement for MultiheadAttention)
        self.attn = MultiHeadGravitationalAttention(
            dim_model=dim_model,
            dim_position=dim_position,
            num_heads=num_heads,
            gravitational_constant=gravitational_constant,
            event_horizon=event_horizon,
            max_force=max_force,
            curvature=curvature,
            different_G_per_head=True,
        )

        # Lightweight feed-forward network (2× expansion, not 4×)
        ff_hidden = int(dim_model * ff_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, dim_model),
            nn.Dropout(dropout),
        )

        # Normalization & residuals
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

        # Optional per-token mass context (informational, affects attention through attn heads)
        if learnable_masses:
            self.token_mass = nn.Parameter(torch.ones(dim_model) * 0.1)
        else:
            self.register_buffer("token_mass", torch.ones(dim_model) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with optional introspection diagnostics.

        Args:
            x: Input tensor ``[batch, seq_len, dim_model]``.
            positions: Precomputed curved/fractal positions ``[seq_len, dim_position]``.
            return_diagnostics: If True, return attention metrics for Mirror Layer.

        Returns:
            output: Transformed tensor ``[batch, seq_len, dim_model]``.
            diagnostics: Optional dict with attention forces, masses, stability.
        """
        seq_len = x.shape[1]

        # === GRAVITATIONAL ATTENTION ===
        attn_output = self.attn(x, positions)  # [batch, seq_len, dim_model]

        # Residual + norm
        x = self.norm1(x + self.dropout(attn_output))

        # === LIGHTWEIGHT FFN ===
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        # === DIAGNOSTICS FOR MIRROR LAYER ===
        diagnostics = None
        if return_diagnostics:
            diag = self.attn.get_attention_diagnostics(x, positions)
            diagnostics = {
                "mean_force": diag.get("head_0", {}).get("mean_force", 0.0),
                "mean_mass": diag.get("head_0", {}).get("mean_mass", 0.0),
                "curvature_active": self.attn.curvature > 0,
                "hawking_limit": self.attn.max_force,
                "seq_len": seq_len,
                "per_head": diag,
            }

        return x, diagnostics


# ---------------------------------------------------------------------------
# Full transformer stack
# ---------------------------------------------------------------------------

class LightweightGravitationalTransformer(nn.Module):
    """
    Complete lightweight transformer stack.

    Designed for:
    - Low-compute inference on edge devices
    - Recursive meta-cognition (Mirror Layer integration)
    - VictorOS cognitive runtime compatibility
    - Fractal attention research prototyping

    Args:
        vocab_size: Vocabulary size for token embedding. ``None`` if the caller
            supplies continuous embeddings directly.
        dim_model: Model (embedding) dimension.
        dim_position: Dimensionality of positional vectors.
        num_layers: Number of gravitational transformer blocks.
        num_heads: Attention heads per block.
        max_seq_len: Maximum supported sequence length.
        curvature: Spacetime curvature for positional embeddings.
        gravitational_constant: Base G (decayed per layer as G * 0.9**i).
        dropout: Dropout probability.
        tie_weights: Tie output projection to embedding weights (autoencoding).
        use_fractal_positions: Use FractalPositionEmbedding instead of
            CurvedPositionEmbedding.
        fractal_dim: Hausdorff dimension for FractalPositionEmbedding.
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        dim_model: int = 128,
        dim_position: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
        curvature: float = 0.15,
        gravitational_constant: float = 1.0,
        dropout: float = 0.1,
        tie_weights: bool = False,
        use_fractal_positions: bool = False,
        fractal_dim: float = 1.5,
    ):
        super().__init__()

        self.dim_model = dim_model
        self.curvature = curvature

        # Optional token embedding layer
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, dim_model)
        else:
            self.embedding = None

        # Positional embedding – curved (default) or fractal
        if use_fractal_positions:
            self.pos_embedding: nn.Module = FractalPositionEmbedding(
                max_seq_len=max_seq_len,
                dim_position=dim_position,
                fractal_dim=fractal_dim,
            )
        else:
            self.pos_embedding = CurvedPositionEmbedding(
                max_seq_len=max_seq_len,
                dim_position=dim_position,
                curvature=curvature,
            )

        # Stack of lightweight gravitational blocks (G decays per layer)
        self.layers = nn.ModuleList([
            LightweightGravitationalBlock(
                dim_model=dim_model,
                dim_position=dim_position,
                num_heads=num_heads,
                curvature=curvature,
                gravitational_constant=gravitational_constant * (0.9 ** i),
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Final normalisation
        self.final_norm = nn.LayerNorm(dim_model)

        # Optional output projection (language modelling head)
        if vocab_size is not None and not tie_weights:
            self.head: Optional[nn.Module] = nn.Linear(dim_model, vocab_size)
        elif tie_weights and self.embedding is not None:
            # Tied weights: reuse embedding matrix (applied as a module wrapper)
            self.head = _TiedHead(self.embedding)
        else:
            self.head = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Xavier initialisation for stable gravitational dynamics."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        mirror_layer_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with optional Mirror Layer integration.

        Args:
            x: Input ``[batch, seq_len, dim_model]`` or token IDs
                ``[batch, seq_len]`` when ``vocab_size`` is set.
            positions: Optional precomputed curved positions.
            return_diagnostics: Enable introspection output.
            mirror_layer_callback: Optional ``callback(layer_idx, diagnostics)``
                for real-time Mirror Layer updates.

        Returns:
            output: Final representations ``[batch, seq_len, dim_model]`` or
                logits ``[batch, seq_len, vocab_size]``.
            diagnostics: Optional aggregated metrics across layers.
        """
        # Embed tokens if needed
        if self.embedding is not None and x.dtype in (torch.long, torch.int):
            x = self.embedding(x)

        # Get curved/fractal positions
        seq_len = x.shape[1]
        if positions is None:
            positions = self.pos_embedding(seq_len)

        all_diagnostics: Optional[List[Dict[str, Any]]] = [] if return_diagnostics else None

        for i, layer in enumerate(self.layers):
            x, layer_diag = layer(x, positions=positions, return_diagnostics=return_diagnostics)

            if return_diagnostics and layer_diag is not None:
                if mirror_layer_callback is not None:
                    mirror_layer_callback(i, layer_diag)
                all_diagnostics.append({"layer": i, **layer_diag})  # type: ignore[union-attr]

        x = self.final_norm(x)

        if self.head is not None:
            x = self.head(x)

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "layers": all_diagnostics,
                "curvature": self.curvature,
                "final_norm_stats": {
                    "mean": x.mean().item(),
                    "std": x.std().item(),
                },
            }

        return x, diagnostics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_attention_snapshot(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Generate a full attention snapshot for Ledger logging or introspection.
        Useful for VictorOS state persistence and causal tracing.
        """
        self.eval()
        with torch.no_grad():
            _, diagnostics = self.forward(x, return_diagnostics=True)
        return {
            "timestamp": (
                torch.cuda.current_stream().cuda_time()
                if torch.cuda.is_available()
                else None
            ),
            "model_config": {
                "dim_model": self.dim_model,
                "curvature": self.curvature,
                "num_layers": len(self.layers),
            },
            "attention_metrics": diagnostics,
        }


# ---------------------------------------------------------------------------
# Helper: tied output head
# ---------------------------------------------------------------------------

class _TiedHead(nn.Module):
    """Output projection sharing weights with an embedding layer."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.embedding.weight.T
