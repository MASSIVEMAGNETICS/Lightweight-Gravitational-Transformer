"""
Tri-Model Architecture
Implements the world / self / environment tri-model fusion core described
in the VictorOS cognitive substrate.

Each sub-model is a full ``LightweightGravitationalTransformer`` with its
own curvature and gravitational constant tuned to its semantic role:

- **WorldModel**: curved semantic space for modelling external context.
  High curvature captures long-range environmental dependencies.
- **SelfModel**: mass-weighted attention for the agent's own internal state.
  Learnable masses track self-relevance of each representation.
- **EnvironmentModel**: force-based urgency for the interaction context.
  Large G emphasises the momentary salience of environmental cues.

The three streams are fused via **cross-gravitational attention**: each
model's output acts as the "position" source for the other two, letting
the models exert gravitational influence on one another.

Architecture
------------
    WorldInput  ──►  WorldModel  ──────────────────────────┐
    SelfInput   ──►  SelfModel   ──► CrossGravitational  ──► FusionHead ──► Output
    EnvInput    ──►  EnvModel    ──────────────────────────┘
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightweight_gravitational_transformer import LightweightGravitationalTransformer


# ---------------------------------------------------------------------------
# Cross-gravitational fusion
# ---------------------------------------------------------------------------

class CrossGravitationalFusion(nn.Module):
    """
    Fuses three streams (world, self, environment) using gravitational cross-
    attention.

    Each stream attends to the other two via gravitational forces computed
    from the streams' mean representations (used as positions on the shared
    manifold).  This lets high-mass, high-relevance streams exert stronger
    pull on the fused output.

    Args:
        dim_model: Common dimensionality of all three streams.
        num_heads: Number of cross-attention heads.
        gravitational_constant: G for cross-stream forces.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int = 4,
        gravitational_constant: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_model = dim_model

        # Standard multi-head cross-attention (world ← self+env, etc.)
        self.world_cross_attn = nn.MultiheadAttention(
            dim_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_cross_attn = nn.MultiheadAttention(
            dim_model, num_heads, dropout=dropout, batch_first=True
        )
        self.env_cross_attn = nn.MultiheadAttention(
            dim_model, num_heads, dropout=dropout, batch_first=True
        )

        # Gravitational mass projections (one per stream)
        self.world_mass = nn.Linear(dim_model, 1, bias=False)
        self.self_mass = nn.Linear(dim_model, 1, bias=False)
        self.env_mass = nn.Linear(dim_model, 1, bias=False)

        self.G = nn.Parameter(torch.tensor(gravitational_constant))

        # Output gate: combine the three cross-attended streams
        self.gate = nn.Linear(dim_model * 3, dim_model)
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        world: torch.Tensor,   # [batch, seq_w, dim]
        self_: torch.Tensor,   # [batch, seq_s, dim]
        env: torch.Tensor,     # [batch, seq_e, dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns fused ``(world_out, self_out, env_out)``, each
        ``[batch, seq, dim]``.
        """
        G = self.G.abs()

        # Compute scalar masses from mean representations
        w_mass = F.softplus(self.world_mass(world.mean(dim=1, keepdim=True)))   # [batch, 1, 1]
        s_mass = F.softplus(self.self_mass(self_.mean(dim=1, keepdim=True)))
        e_mass = F.softplus(self.env_mass(env.mean(dim=1, keepdim=True)))

        # Scale key/values by gravitational mass before cross-attention
        w_scaled = world * (G * w_mass)
        s_scaled = self_ * (G * s_mass)
        e_scaled = env * (G * e_mass)

        # World attends to self + env
        kv_world = torch.cat([s_scaled, e_scaled], dim=1)
        world_out, _ = self.world_cross_attn(world, kv_world, kv_world)

        # Self attends to world + env
        kv_self = torch.cat([w_scaled, e_scaled], dim=1)
        self_out, _ = self.self_cross_attn(self_, kv_self, kv_self)

        # Env attends to world + self
        kv_env = torch.cat([w_scaled, s_scaled], dim=1)
        env_out, _ = self.env_cross_attn(env, kv_env, kv_env)

        return world_out, self_out, env_out


# ---------------------------------------------------------------------------
# Tri-Model Transformer
# ---------------------------------------------------------------------------

class TriModelTransformer(nn.Module):
    """
    Tri-model cognitive architecture: WorldModel + SelfModel + EnvironmentModel.

    The three LGT sub-models process their respective input streams in
    parallel, then cross-attend via ``CrossGravitationalFusion``.  The
    fused representations are pooled and projected to a common output space.

    Each sub-model has tuned hyper-parameters reflecting its semantic role:

    +--------------+------------+-------+-------+
    | Sub-model    | Curvature  |   G   | Role  |
    +==============+============+=======+=======+
    | WorldModel   | high (0.25)| 1.0   | External context |
    | SelfModel    | medium (0.15)| 0.8 | Internal state   |
    | EnvironmentModel | low (0.10)| 1.2 | Interaction urgency |
    +--------------+------------+-------+-------+

    Args:
        dim_model: Shared model dimension for all three sub-models.
        dim_position: Positional vector dimension.
        num_layers: Depth of each sub-model (same for all three).
        num_heads: Attention heads per block.
        vocab_size: Optional vocabulary size (set if inputs are token IDs).
        max_seq_len: Maximum sequence length per stream.
        dropout: Dropout probability.
        use_fractal_positions: Enable fractal position embeddings.
        output_dim: Output projection dimension. Defaults to ``dim_model``.
    """

    def __init__(
        self,
        dim_model: int = 128,
        dim_position: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_fractal_positions: bool = False,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        self.dim_model = dim_model
        output_dim = output_dim or dim_model

        # Sub-models output dim_model-dimensional representations (no vocab head).
        # Vocabulary embedding lives here at the TriModel level so that each
        # sub-model receives continuous embeddings, not raw token IDs.
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, dim_model)
        else:
            self.embedding = None

        sub_kwargs = dict(
            vocab_size=None,          # No head on sub-models; we handle embeddings here
            dim_model=dim_model,
            dim_position=dim_position,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_fractal_positions=use_fractal_positions,
        )

        # Three specialised sub-models
        self.world_model = LightweightGravitationalTransformer(
            curvature=0.25,
            gravitational_constant=1.0,
            **sub_kwargs,
        )
        self.self_model = LightweightGravitationalTransformer(
            curvature=0.15,
            gravitational_constant=0.8,
            **sub_kwargs,
        )
        self.env_model = LightweightGravitationalTransformer(
            curvature=0.10,
            gravitational_constant=1.2,
            **sub_kwargs,
        )

        # Cross-gravitational fusion layer
        self.fusion = CrossGravitationalFusion(
            dim_model=dim_model,
            num_heads=num_heads,
            gravitational_constant=1.0,
            dropout=dropout,
        )

        # Final projection – input is the concatenation of 3 streams
        self.final_norm = nn.LayerNorm(dim_model * 3)
        self.output_proj = nn.Linear(dim_model * 3, output_dim)

    def forward(
        self,
        world_input: torch.Tensor,
        self_input: torch.Tensor,
        env_input: torch.Tensor,
        return_diagnostics: bool = False,
        mirror_layer_callback: Optional[Callable[[str, int, Dict[str, Any]], None]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass over all three streams.

        Args:
            world_input: External context tokens/embeddings ``[batch, seq_w, dim]``
                or token IDs ``[batch, seq_w]`` when ``vocab_size`` is set.
            self_input: Agent internal-state tokens ``[batch, seq_s, dim]``.
            env_input: Interaction context tokens ``[batch, seq_e, dim]``.
            return_diagnostics: Include per-stream diagnostics.
            mirror_layer_callback: Optional ``callback(stream_name, layer_idx, diag)``
                for Mirror Layer integration.  Receives the stream name as first arg.

        Returns:
            output: ``[batch, max_seq_len, output_dim]`` fused representation.
            diagnostics: Optional dict with per-stream and fusion diagnostics.
        """
        # Apply shared embedding if inputs are token IDs
        if self.embedding is not None:
            def _maybe_embed(inp: torch.Tensor) -> torch.Tensor:
                if inp.dtype in (torch.long, torch.int):
                    return self.embedding(inp)
                return inp
            world_input = _maybe_embed(world_input)
            self_input = _maybe_embed(self_input)
            env_input = _maybe_embed(env_input)

        # Build per-stream Mirror Layer callbacks
        def _make_cb(name: str) -> Optional[Callable]:
            if mirror_layer_callback is None:
                return None
            return lambda idx, diag: mirror_layer_callback(name, idx, diag)

        # Forward each sub-model
        world_out, world_diag = self.world_model(
            world_input, return_diagnostics=return_diagnostics,
            mirror_layer_callback=_make_cb("world"),
        )
        self_out, self_diag = self.self_model(
            self_input, return_diagnostics=return_diagnostics,
            mirror_layer_callback=_make_cb("self"),
        )
        env_out, env_diag = self.env_model(
            env_input, return_diagnostics=return_diagnostics,
            mirror_layer_callback=_make_cb("env"),
        )

        # Align sequence lengths via padding (broadcast to max length)
        max_len = max(world_out.shape[1], self_out.shape[1], env_out.shape[1])
        world_out = _pad_seq(world_out, max_len)
        self_out = _pad_seq(self_out, max_len)
        env_out = _pad_seq(env_out, max_len)

        # Cross-gravitational fusion
        world_fused, self_fused, env_fused = self.fusion(world_out, self_out, env_out)

        # Concatenate all three fused streams, normalise, then project
        fused = torch.cat([world_fused, self_fused, env_fused], dim=-1)  # [batch, seq, 3*dim]
        fused_normed = self.final_norm(fused)
        output = self.output_proj(fused_normed)

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "world": world_diag,
                "self": self_diag,
                "env": env_diag,
                "fusion": {
                    "world_G": float(self.fusion.G.abs().detach()),
                    "fused_mean": float(output.detach().mean()),
                    "fused_std": float(output.detach().std()),
                },
            }

        return output, diagnostics

    def get_tri_snapshot(
        self,
        world_input: torch.Tensor,
        self_input: torch.Tensor,
        env_input: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Generate a combined attention snapshot across all three sub-models.
        Suitable for VictorOS Ledger causal tracing.
        """
        self.eval()
        with torch.no_grad():
            _, diagnostics = self.forward(
                world_input, self_input, env_input, return_diagnostics=True
            )
        return {
            "world_snapshot": self.world_model.get_attention_snapshot(world_input),
            "self_snapshot": self.self_model.get_attention_snapshot(self_input),
            "env_snapshot": self.env_model.get_attention_snapshot(env_input),
            "fusion_diagnostics": diagnostics.get("fusion") if diagnostics else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Zero-pad sequence dimension to ``target_len``."""
    current = x.shape[1]
    if current >= target_len:
        return x[:, :target_len, :]
    pad = torch.zeros(x.shape[0], target_len - current, x.shape[2], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)
