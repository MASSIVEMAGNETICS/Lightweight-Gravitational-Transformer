"""
Edge Model Export Example
==========================
Demonstrates how to export LGT models for edge deployment using the four
preset configurations, with optional quantisation.

Run from the repository root:
    python examples/edge_export.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from export_edge_model import (
    PRESETS,
    build_model,
    export_edge_model,
    quantize_dynamic,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Example 1: Inspect available presets
# ---------------------------------------------------------------------------

def presets_example():
    print("=" * 60)
    print("Example 1: Available Presets")
    print("=" * 60)

    print(f"{'Preset':<15} {'dim_model':<12} {'layers':<8} {'heads':<8} {'curvature':<10}")
    print("-" * 53)
    for name, cfg in PRESETS.items():
        print(f"{name:<15} {cfg['dim_model']:<12} {cfg['num_layers']:<8} "
              f"{cfg['num_heads']:<8} {cfg['curvature']:<10}")


# ---------------------------------------------------------------------------
# Example 2: Build and inspect model sizes
# ---------------------------------------------------------------------------

def model_sizes_example():
    print()
    print("=" * 60)
    print("Example 2: Model Parameter Counts")
    print("=" * 60)

    for preset_name in PRESETS:
        model = build_model(config_name=preset_name, vocab_size=32000)
        n_params = sum(p.numel() for p in model.parameters())
        mem_fp32 = n_params * 4 / (1024 ** 2)
        print(f"{preset_name:<15} {n_params:>10,} params   {mem_fp32:.2f} MB (FP32)")


# ---------------------------------------------------------------------------
# Example 3: Quantisation comparison
# ---------------------------------------------------------------------------

def quantisation_example():
    print()
    print("=" * 60)
    print("Example 3: Quantisation (edge_150k preset)")
    print("=" * 60)

    model_fp32 = build_model("edge_150k", vocab_size=1000)
    n_params = sum(p.numel() for p in model_fp32.parameters())
    mem_fp32 = n_params * 4 / (1024 ** 2)
    print(f"FP32:   {n_params:,} params, {mem_fp32:.3f} MB")

    # FP16
    model_fp16 = build_model("edge_150k", vocab_size=1000)
    model_fp16 = quantize_dynamic(model_fp16, dtype="float16")
    # FP16 roughly halves memory
    print(f"FP16:   {n_params:,} params, ~{mem_fp32/2:.3f} MB (estimated)")

    # INT8
    model_int8 = build_model("edge_150k", vocab_size=1000)
    model_int8 = quantize_dynamic(model_int8, dtype="int8")
    print(f"INT8:   {n_params:,} params, ~{mem_fp32/4:.3f} MB (estimated, linear layers only)")

    # Run inference to verify quantised models work
    x = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        out_fp32 = model_fp32(x)[0]
        out_fp16 = model_fp16(x.to(model_fp16.embedding.weight.device))[0]
    print(f"FP32 output shape: {out_fp32.shape}")
    print(f"FP16 output shape: {out_fp16.shape}")


# ---------------------------------------------------------------------------
# Example 4: Full export pipeline (to temp directory)
# ---------------------------------------------------------------------------

def full_export_example():
    print()
    print("=" * 60)
    print("Example 4: Full Export Pipeline")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_edge_model(
            config_name="edge_150k",
            vocab_size=1000,
            max_seq_len=128,
            quantize="none",
            output_dir=tmpdir,
            use_fractal_positions=False,
            example_seq_len=16,
        )

        print(f"Checkpoint  : {os.path.basename(paths['checkpoint'])}")
        if paths["torchscript"]:
            print(f"TorchScript : {os.path.basename(paths['torchscript'])}")

        # Inspect the checkpoint
        state = torch.load(paths["checkpoint"], weights_only=False)
        print(f"Metadata    : {state['metadata']}")

        # Load checkpoint and run inference
        model = build_model(
            config_name=state["metadata"]["config"],
            vocab_size=state["metadata"]["vocab_size"],
            max_seq_len=state["metadata"]["max_seq_len"],
        )
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        x = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            out, _ = model(x)
        print(f"Loaded model output shape: {out.shape}")

        # Load TorchScript model (may have been skipped for this model type)
        if paths["torchscript"] and os.path.exists(paths["torchscript"]):
            scripted = torch.jit.load(paths["torchscript"])
            with torch.no_grad():
                ts_out = scripted(x)
            print(f"TorchScript output shape : {ts_out[0].shape}")
        else:
            print("TorchScript export was skipped (not available for this model config)")


# ---------------------------------------------------------------------------
# Example 5: Save a custom checkpoint
# ---------------------------------------------------------------------------

def custom_checkpoint_example():
    print()
    print("=" * 60)
    print("Example 5: Custom Checkpoint Save/Load")
    print("=" * 60)

    model = build_model("meta_probe", vocab_size=500)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "custom.pt")
        saved_path = save_checkpoint(
            model,
            ckpt_path,
            metadata={"experiment": "meta_probe_demo", "epoch": 5},
        )
        print(f"Saved to: {os.path.basename(saved_path)}")

        state = torch.load(saved_path, weights_only=False)
        print(f"Metadata: {state['metadata']}")
        print(f"State dict keys: {list(state['model_state_dict'].keys())[:3]} …")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    presets_example()
    model_sizes_example()
    quantisation_example()
    full_export_example()
    custom_checkpoint_example()
    print("\nAll edge export examples completed ✓")
