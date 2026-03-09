"""
LGT Benchmarking Suite
Measures memory footprint, inference latency, and throughput of the
Lightweight Gravitational Transformer across all size presets.

Run directly:
    python benchmarks/benchmark_lgt.py
    python benchmarks/benchmark_lgt.py --config edge_150k --seq-len 64 --runs 100
"""

import argparse
import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from export_edge_model import PRESETS, build_model


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _model_size_mb(model: nn.Module) -> float:
    """Estimate FP32 model size in MB (parameters only)."""
    return _count_params(model) * 4 / (1024 ** 2)


def _process_memory_mb() -> float:
    """Current process RSS in MB (Linux only; returns 0 elsewhere)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def measure_latency(
    model: nn.Module,
    example_input: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Measure per-call inference latency statistics (ms).

    Returns dict with ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``,
    ``median_ms``, and ``p95_ms``.
    """
    model.eval()
    model.to(device)
    example_input = example_input.to(device)

    with torch.no_grad():
        # Warm-up passes (JIT compilation, caches, etc.)
        for _ in range(num_warmup):
            _ = model(example_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(example_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)
    return {
        "mean_ms": sum(latencies) / n,
        "std_ms": (sum((x - sum(latencies) / n) ** 2 for x in latencies) / n) ** 0.5,
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "median_ms": latencies[n // 2],
        "p95_ms": latencies[int(n * 0.95)],
    }


def measure_throughput(
    model: nn.Module,
    example_input: torch.Tensor,
    duration_s: float = 3.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Measure throughput: inferences per second and tokens per second.

    Returns dict with ``inferences_per_sec`` and ``tokens_per_sec``.
    """
    model.eval()
    model.to(device)
    example_input = example_input.to(device)

    batch_size = example_input.shape[0]
    seq_len = example_input.shape[1] if example_input.dim() > 1 else 1

    count = 0
    with torch.no_grad():
        # Warm up
        _ = model(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            _ = model(example_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            count += 1

    inferences_per_sec = count / duration_s
    tokens_per_sec = inferences_per_sec * batch_size * seq_len
    return {
        "inferences_per_sec": inferences_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "total_inferences": count,
    }


def measure_memory_footprint(
    model: nn.Module,
    example_input: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Estimate memory usage during inference.

    Returns ``model_mb`` (parameter bytes), ``activation_mb`` (estimated),
    ``total_mb``.  For CUDA devices, reports ``cuda_allocated_mb``.
    """
    model.eval()
    model.to(device)

    model_mb = _model_size_mb(model)

    # Estimate activation memory: O(batch * seq * dim * num_layers)
    batch = example_input.shape[0]
    seq = example_input.shape[1] if example_input.dim() > 1 else 1
    dim = getattr(model, "dim_model", 128)
    layers = len(getattr(model, "layers", []))
    activation_mb = batch * seq * dim * layers * 4 / (1024 ** 2)

    cuda_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(example_input.to(device))
        cuda_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "model_mb": model_mb,
        "activation_mb_est": activation_mb,
        "total_mb_est": model_mb + activation_mb,
        "cuda_peak_mb": cuda_mb,
    }


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def benchmark_preset(
    config_name: str,
    vocab_size: int = 32000,
    seq_len: int = 64,
    batch_size: int = 1,
    num_runs: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Run a complete benchmark for a single preset configuration."""
    gc.collect()

    model = build_model(config_name, vocab_size=vocab_size)
    example = torch.randint(0, vocab_size, (batch_size, seq_len))

    params = _count_params(model)
    memory = measure_memory_footprint(model, example, device)
    latency = measure_latency(model, example, num_warmup=5, num_runs=num_runs, device=device)
    throughput = measure_throughput(model, example, duration_s=2.0, device=device)

    return {
        "config": config_name,
        "params": params,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "device": str(device),
        "memory": memory,
        "latency": latency,
        "throughput": throughput,
    }


def run_all_benchmarks(
    seq_len: int = 64,
    batch_size: int = 1,
    num_runs: int = 50,
    configs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Benchmark all (or a subset of) presets and return results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = configs or list(PRESETS.keys())
    results = []
    for name in configs:
        print(f"\n─── Benchmarking: {name} ───")
        result = benchmark_preset(
            name,
            seq_len=seq_len,
            batch_size=batch_size,
            num_runs=num_runs,
            device=device,
        )
        results.append(result)
        _print_result(result)
    return results


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _print_result(r: Dict[str, Any]) -> None:
    m = r["memory"]
    l = r["latency"]
    t = r["throughput"]
    print(f"  Parameters      : {r['params']:>10,}")
    print(f"  Model size (FP32): {m['model_mb']:>7.2f} MB")
    print(f"  Total mem est   : {m['total_mb_est']:>7.2f} MB")
    print(f"  Latency  mean   : {l['mean_ms']:>7.2f} ms")
    print(f"  Latency  p95    : {l['p95_ms']:>7.2f} ms")
    print(f"  Throughput      : {t['inferences_per_sec']:>7.1f} inf/s  |  {t['tokens_per_sec']:>10.0f} tok/s")


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print a compact comparison table."""
    header = f"{'Config':<16} {'Params':>10} {'Size MB':>8} {'Lat ms':>8} {'Inf/s':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r['config']:<16} {r['params']:>10,} "
            f"{r['memory']['model_mb']:>8.2f} "
            f"{r['latency']['mean_ms']:>8.2f} "
            f"{r['throughput']['inferences_per_sec']:>8.1f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark LGT presets.")
    p.add_argument("--config", nargs="*", default=None,
                   help="Preset name(s) to benchmark (default: all).")
    p.add_argument("--seq-len", type=int, default=64, metavar="N")
    p.add_argument("--batch-size", type=int, default=1, metavar="N")
    p.add_argument("--runs", type=int, default=50, metavar="N",
                   help="Number of timed inference runs (default: 50).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = run_all_benchmarks(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_runs=args.runs,
        configs=args.config,
    )
    print_summary_table(results)
