"""
Edge Model Export
Export and quantise the Lightweight Gravitational Transformer for deployment
on resource-constrained hardware (e.g., Raspberry Pi 4).

Supports:
- TorchScript tracing for portability and inference-time optimisation.
- Dynamic INT8 quantisation for ~4× size reduction and faster CPU inference.
- FP16 conversion for moderate-memory savings.
- CLI interface: ``python export_edge_model.py --config edge_150k --quantize int8``

Preset configurations
---------------------
edge_150k   : dim=64,  layers=2, heads=2  (~150K params, <1 MB FP32)
meta_probe  : dim=128, layers=4, heads=4  (~600K params)
fractal_res : dim=256, layers=6, heads=8  (~2.1M params)
victorcos   : dim=192, layers=5, heads=6  (~1.4M params)
"""

import argparse
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from lightweight_gravitational_transformer import LightweightGravitationalTransformer


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "edge_150k": dict(dim_model=64,  dim_position=32,  num_layers=2, num_heads=2,  curvature=0.10),
    "meta_probe": dict(dim_model=128, dim_position=64,  num_layers=4, num_heads=4,  curvature=0.15),
    "fractal_res": dict(dim_model=256, dim_position=128, num_layers=6, num_heads=8,  curvature=0.25),
    "victorcos":  dict(dim_model=192, dim_position=96,  num_layers=5, num_heads=6,  curvature=0.18),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_memory_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    bytes_per_elem = {torch.float32: 4, torch.float16: 2, torch.int8: 1}.get(dtype, 4)
    return _count_params(model) * bytes_per_elem / (1024 ** 2)


# ---------------------------------------------------------------------------
# Build model from preset / kwargs
# ---------------------------------------------------------------------------

def build_model(
    config_name: str = "edge_150k",
    vocab_size: int = 32000,
    max_seq_len: int = 512,
    use_fractal_positions: bool = False,
    **kwargs: Any,
) -> LightweightGravitationalTransformer:
    """Build an LGT from a named preset, with optional overrides."""
    preset = dict(PRESETS.get(config_name, PRESETS["edge_150k"]))
    preset.update(kwargs)
    model = LightweightGravitationalTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        use_fractal_positions=use_fractal_positions,
        **preset,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_torchscript(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str,
) -> str:
    """
    Trace the model with TorchScript and save to ``output_path``.

    Returns the saved path.
    """
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_input,), strict=False)
    torch.jit.save(traced, output_path)
    return output_path


def quantize_dynamic(
    model: nn.Module,
    dtype: str = "int8",
) -> nn.Module:
    """
    Apply dynamic quantisation to all ``nn.Linear`` layers.

    Args:
        model: The model to quantise (modified in-place copy returned).
        dtype: ``"int8"`` (default) or ``"float16"``.

    Returns:
        Quantised model.
    """
    if dtype == "float16":
        return model.half()

    q_dtype = torch.qint8
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=q_dtype,
    )
    return quantized


def save_checkpoint(
    model: nn.Module,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model weights + metadata as a ``torch.save`` checkpoint."""
    state = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
        "exported_at": time.time(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(state, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Full export pipeline
# ---------------------------------------------------------------------------

def export_edge_model(
    config_name: str = "edge_150k",
    vocab_size: int = 32000,
    max_seq_len: int = 512,
    quantize: str = "none",
    output_dir: str = "exported_models",
    use_fractal_positions: bool = False,
    example_seq_len: int = 64,
) -> Dict[str, str]:
    """
    Full export pipeline: build → (quantise) → TorchScript → save.

    Args:
        config_name: Preset name (see ``PRESETS``).
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length the exported model supports.
        quantize: Quantisation mode – ``"none"``, ``"int8"``, or ``"float16"``.
        output_dir: Directory to write exported artefacts.
        use_fractal_positions: Enable fractal position embeddings.
        example_seq_len: Sequence length used for TorchScript tracing.

    Returns:
        Dict of ``{"checkpoint": path, "torchscript": path, "config": ...}``.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[export] Building model: {config_name}")
    model = build_model(
        config_name=config_name,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        use_fractal_positions=use_fractal_positions,
    )
    n_params = _count_params(model)
    mem_fp32 = _estimate_memory_mb(model, torch.float32)
    print(f"[export]  Parameters : {n_params:,}")
    print(f"[export]  Memory (FP32): {mem_fp32:.2f} MB")

    # Quantise
    if quantize != "none":
        print(f"[export] Quantising ({quantize}) …")
        model = quantize_dynamic(model, dtype=quantize)
        dtype_map = {"int8": torch.qint8, "float16": torch.float16}
        q_dtype = dtype_map.get(quantize, torch.float32)
        mem_q = _estimate_memory_mb(model, q_dtype)
        print(f"[export]  Memory ({quantize}): {mem_q:.2f} MB")

    # Save raw checkpoint
    suffix = f"_{quantize}" if quantize != "none" else ""
    ckpt_path = os.path.join(output_dir, f"lgt_{config_name}{suffix}.pt")
    metadata: Dict[str, Any] = {
        "config": config_name,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "quantize": quantize,
        "n_params": n_params,
        "use_fractal_positions": use_fractal_positions,
    }
    save_checkpoint(model, ckpt_path, metadata)
    print(f"[export] Checkpoint saved → {ckpt_path}")

    # TorchScript export (skip for quantised INT8 as it requires scripting)
    ts_path = os.path.join(output_dir, f"lgt_{config_name}{suffix}_traced.pt")
    example = torch.randint(0, vocab_size, (1, example_seq_len))
    try:
        # For quantised models, fall back to script instead of trace
        if quantize == "int8":
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, ts_path)
        else:
            export_torchscript(model, example, ts_path)
        print(f"[export] TorchScript saved → {ts_path}")
    except Exception as exc:
        print(f"[export] TorchScript export skipped ({exc}); checkpoint only.")
        ts_path = ""

    return {"checkpoint": ckpt_path, "torchscript": ts_path, "config": metadata}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a Lightweight Gravitational Transformer for edge deployment."
    )
    p.add_argument(
        "--config",
        default="edge_150k",
        choices=list(PRESETS.keys()),
        help="Model size preset (default: edge_150k).",
    )
    p.add_argument("--vocab-size", type=int, default=32000, metavar="N")
    p.add_argument("--max-seq-len", type=int, default=512, metavar="N")
    p.add_argument(
        "--quantize",
        default="none",
        choices=["none", "int8", "float16"],
        help="Quantisation mode (default: none).",
    )
    p.add_argument("--output-dir", default="exported_models", metavar="DIR")
    p.add_argument(
        "--fractal-positions",
        action="store_true",
        help="Use FractalPositionEmbedding instead of CurvedPositionEmbedding.",
    )
    p.add_argument(
        "--example-seq-len",
        type=int,
        default=64,
        metavar="N",
        help="Sequence length for TorchScript tracing (default: 64).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    paths = export_edge_model(
        config_name=args.config,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        quantize=args.quantize,
        output_dir=args.output_dir,
        use_fractal_positions=args.fractal_positions,
        example_seq_len=args.example_seq_len,
    )
    print("\n[export] Done.")
    print(f"  Checkpoint  : {paths['checkpoint']}")
    print(f"  TorchScript : {paths['torchscript'] or '(skipped)'}")
