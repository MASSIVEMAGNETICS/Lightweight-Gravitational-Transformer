"""
Basic Inference Example
=======================
Demonstrates minimal forward pass with continuous embeddings and token IDs.
Run from the repository root:
    python examples/basic_inference.py
"""

import sys
import os

# Allow running from repository root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer


def continuous_embedding_example():
    print("=" * 60)
    print("Example 1: Continuous Embedding Input")
    print("=" * 60)

    model = LightweightGravitationalTransformer(
        dim_model=128,
        dim_position=64,
        num_layers=4,
        num_heads=4,
        max_seq_len=512,
        curvature=0.15,
        dropout=0.0,  # no dropout at inference time
    )
    model.eval()

    # Batch of 2 sequences, length 32, embedding dim 128
    x = torch.randn(2, 32, 128)

    with torch.no_grad():
        output, diagnostics = model(x, return_diagnostics=True)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Curvature    : {diagnostics['curvature']}")
    print(f"Num layers   : {len(diagnostics['layers'])}")

    layer0 = diagnostics["layers"][0]
    print(f"Layer 0 mean_force : {layer0['mean_force']:.6f}")
    print(f"Layer 0 mean_mass  : {layer0['mean_mass']:.6f}")


def language_model_example():
    print()
    print("=" * 60)
    print("Example 2: Language Model (Token IDs)")
    print("=" * 60)

    vocab_size = 1000
    model = LightweightGravitationalTransformer(
        vocab_size=vocab_size,
        dim_model=64,
        dim_position=32,
        num_layers=2,
        num_heads=2,
        max_seq_len=128,
        tie_weights=True,  # share embedding and output-projection weights
        dropout=0.0,
    )
    model.eval()

    # Token IDs: [batch=4, seq=16]
    token_ids = torch.randint(0, vocab_size, (4, 16))

    with torch.no_grad():
        logits, _ = model(token_ids)

    print(f"Token IDs shape : {token_ids.shape}")
    print(f"Logits shape    : {logits.shape}")  # [4, 16, 1000]

    # Greedy decode
    predicted = logits.argmax(dim=-1)
    print(f"Predicted IDs   : {predicted[0].tolist()}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")


def fractal_position_example():
    print()
    print("=" * 60)
    print("Example 3: Fractal Position Embeddings")
    print("=" * 60)

    model = LightweightGravitationalTransformer(
        dim_model=128,
        dim_position=64,
        num_layers=4,
        num_heads=4,
        use_fractal_positions=True,
        fractal_dim=1.5,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(1, 64, 128)
    with torch.no_grad():
        output, diag = model(x, return_diagnostics=True)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Curvature    : {diag['curvature']}")


def custom_positions_example():
    print()
    print("=" * 60)
    print("Example 4: Precomputed Custom Positions")
    print("=" * 60)

    model = LightweightGravitationalTransformer(dim_model=128, dim_position=64)
    model.eval()

    x = torch.randn(1, 16, 128)
    # Supply your own positional geometry
    custom_positions = torch.randn(16, 64)

    with torch.no_grad():
        output, _ = model(x, positions=custom_positions)

    print(f"Input  shape            : {x.shape}")
    print(f"Custom positions shape  : {custom_positions.shape}")
    print(f"Output shape            : {output.shape}")


def attention_snapshot_example():
    print()
    print("=" * 60)
    print("Example 5: Attention Snapshot")
    print("=" * 60)

    model = LightweightGravitationalTransformer(dim_model=64, num_layers=2)
    model.eval()

    x = torch.randn(1, 8, 64)
    snapshot = model.get_attention_snapshot(x)

    print("Model config   :", snapshot["model_config"])
    print("Timestamp      :", snapshot["timestamp"])

    layers = snapshot["attention_metrics"]["layers"]
    for layer in layers:
        print(f"  Layer {layer['layer']}: force={layer['mean_force']:.4f}, "
              f"mass={layer['mean_mass']:.4f}")


if __name__ == "__main__":
    continuous_embedding_example()
    language_model_example()
    fractal_position_example()
    custom_positions_example()
    attention_snapshot_example()
    print("\nAll basic inference examples completed ✓")
