"""
Tri-Model Fusion Example
=========================
Demonstrates the TriModelTransformer world/self/environment fusion
architecture with Mirror Layer integration.

Run from the repository root:
    python examples/tri_model_fusion.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tri_model import TriModelTransformer
from victorcos_module import Ledger, MirrorLayer


# ---------------------------------------------------------------------------
# Example 1: Basic forward pass
# ---------------------------------------------------------------------------

def basic_tri_model_example():
    print("=" * 60)
    print("Example 1: Basic Tri-Model Forward Pass")
    print("=" * 60)

    model = TriModelTransformer(
        dim_model=64,
        dim_position=32,
        num_layers=2,
        num_heads=2,
        vocab_size=None,   # continuous embeddings
        max_seq_len=64,
        output_dim=64,
    )
    model.eval()

    # Three input streams (different sequence lengths are supported)
    world = torch.randn(2, 16, 64)   # external context
    self_ = torch.randn(2, 8,  64)   # internal state
    env   = torch.randn(2, 4,  64)   # interaction urgency

    with torch.no_grad():
        output, diagnostics = model(world, self_, env, return_diagnostics=True)

    print(f"World input  : {world.shape}")
    print(f"Self input   : {self_.shape}")
    print(f"Env input    : {env.shape}")
    print(f"Output       : {output.shape}")
    print(f"Fusion G     : {diagnostics['fusion']['world_G']:.4f}")
    print(f"Fused mean   : {diagnostics['fusion']['fused_mean']:.4f}")
    print(f"Fused std    : {diagnostics['fusion']['fused_std']:.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {n_params:,}")


# ---------------------------------------------------------------------------
# Example 2: Token ID inputs
# ---------------------------------------------------------------------------

def token_id_example():
    print()
    print("=" * 60)
    print("Example 2: Token ID Inputs (shared embedding)")
    print("=" * 60)

    VOCAB = 1000
    model = TriModelTransformer(
        dim_model=64,
        num_layers=2,
        num_heads=2,
        vocab_size=VOCAB,
        max_seq_len=64,
    )
    model.eval()

    world = torch.randint(0, VOCAB, (1, 16))
    self_ = torch.randint(0, VOCAB, (1, 8))
    env   = torch.randint(0, VOCAB, (1, 4))

    with torch.no_grad():
        output, _ = model(world, self_, env)

    print(f"Token ID world  : {world.shape}")
    print(f"Token ID self   : {self_.shape}")
    print(f"Token ID env    : {env.shape}")
    print(f"Output          : {output.shape}")


# ---------------------------------------------------------------------------
# Example 3: Mirror Layer callback
# ---------------------------------------------------------------------------

def mirror_layer_example():
    print()
    print("=" * 60)
    print("Example 3: Mirror Layer Callback")
    print("=" * 60)

    model = TriModelTransformer(
        dim_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
    )
    model.eval()

    ledger = Ledger(agent_id="tri_mirror")
    mirror = MirrorLayer(ledger=ledger, max_force_threshold=40.0)

    # The tri-model callback receives (stream_name, layer_idx, diag)
    stream_events = []

    def tri_callback(stream_name: str, layer_idx: int, diag: dict):
        stream_events.append({"stream": stream_name, "layer": layer_idx})
        mirror(layer_idx, diag)

    world = torch.randn(1, 16, 64)
    self_ = torch.randn(1, 8,  64)
    env   = torch.randn(1, 4,  64)

    with torch.no_grad():
        output, _ = model(
            world, self_, env,
            return_diagnostics=True,
            mirror_layer_callback=tri_callback,
        )

    print(f"Stream events received : {len(stream_events)}")
    for ev in stream_events[:6]:
        print(f"  {ev['stream']:<8} layer={ev['layer']}")
    if len(stream_events) > 6:
        print(f"  … ({len(stream_events) - 6} more)")

    print(f"Stability score        : {mirror.stability_score():.4f}")
    print(f"Ledger entries         : {len(ledger)}")


# ---------------------------------------------------------------------------
# Example 4: Causal trace snapshot
# ---------------------------------------------------------------------------

def snapshot_example():
    print()
    print("=" * 60)
    print("Example 4: VictorOS Causal Trace Snapshot")
    print("=" * 60)

    model = TriModelTransformer(
        dim_model=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
    )
    model.eval()

    world = torch.randn(1, 8, 64)
    self_ = torch.randn(1, 4, 64)
    env   = torch.randn(1, 4, 64)

    snapshot = model.get_tri_snapshot(world, self_, env)

    print("Snapshot keys:", list(snapshot.keys()))
    print("World snapshot config:", snapshot["world_snapshot"]["model_config"])
    if snapshot["fusion_diagnostics"]:
        print("Fusion G:", snapshot["fusion_diagnostics"]["world_G"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    basic_tri_model_example()
    token_id_example()
    mirror_layer_example()
    snapshot_example()
    print("\nAll tri-model fusion examples completed ✓")
