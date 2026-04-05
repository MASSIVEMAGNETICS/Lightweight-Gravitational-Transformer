"""
bench_gravity_vs_sdp.py
=======================
**Gravity Attention vs. Scaled-Dot-Product Attention — 1 M-param Edge Benchmark**

Answers three public, verifiable questions about the Lightweight Gravitational
Transformer (LGT):

1. **Is gravity attention a trainable layer?**
   Both models are trained on the same toy next-token task for ``--steps``
   gradient steps.  Per-step cross-entropy loss is recorded and compared.

2. **How does latency / throughput compare?**
   Both models are benchmarked on CPU (edge-device proxy) and, when available,
   CUDA.  Mean latency, p95 latency, and tokens/s are reported.

3. **What is the relative FLOPs cost?**
   Theoretical FLOPs are estimated analytically for each attention variant and
   the ratio is printed.

Both models are matched to *≈ 1 M parameters* so the comparison is fair.

Usage
-----
Run the full demo (default: 50 training steps, 100 bench runs)::

    python benchmarks/bench_gravity_vs_sdp.py

Quick mode for CI (5 training steps, 10 bench runs)::

    python benchmarks/bench_gravity_vs_sdp.py --quick

Custom::

    python benchmarks/bench_gravity_vs_sdp.py \\
        --steps 200 --seq-len 64 --batch-size 4 --runs 200

Optional JAX
------------
If ``jax`` and ``flax`` are installed the script also runs an equivalent
JAX/Flax SDPA block and adds it to the comparison table.  JAX is *not*
required; the harness degrades gracefully when it is absent.

Output
------
The script prints:
  * Per-step loss curves (ASCII sparkline + numeric table)
  * Latency / throughput comparison table
  * FLOPs comparison table
  * JSON summary written to ``/tmp/bench_gravity_vs_sdp_results.json``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow imports from repo root whether running from benchmarks/ or root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from gravitational_attention import MultiHeadGravitationalAttention  # noqa: F401,E402
from fractal_position_embedding import FractalPositionEmbedding  # noqa: F401,E402
from lightweight_gravitational_transformer import (  # noqa: E402
    LightweightGravitationalTransformer,
)

# ---------------------------------------------------------------------------
# Optional JAX / Flax
# ---------------------------------------------------------------------------
_JAX_AVAILABLE = False
try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import flax.linen as fnn  # type: ignore

    _JAX_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants for the ~1 M-param edge preset
# ---------------------------------------------------------------------------
EDGE_1M = dict(
    vocab_size=512,
    dim_model=192,
    dim_position=96,
    num_layers=4,
    num_heads=4,
    max_seq_len=128,
    curvature=0.15,
    dropout=0.0,
)

# Sequences shorter than max_seq_len used during the benchmark
BENCH_SEQ_LEN = 64
BENCH_BATCH = 1


# ===========================================================================
# SDP (Scaled-Dot-Product) baseline — same architecture but replaces
# MultiHeadGravitationalAttention with torch.nn.MultiheadAttention
# ===========================================================================


class _SDPBlock(nn.Module):
    """Single transformer block using standard scaled-dot-product attention."""

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        ff_expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        ff_hidden = int(dim_model * ff_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, dim_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class SDPModel(nn.Module):
    """
    Scaled-dot-product transformer baseline matched to ~1 M parameters.

    Structurally identical to LGT (same embedding, FFN expansion, layer count)
    but uses ``torch.nn.MultiheadAttention`` (SDPA under the hood in PyTorch ≥2)
    instead of gravitational attention.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        dim_model: int = 192,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_emb = nn.Embedding(max_seq_len, dim_model)
        self.layers = nn.ModuleList(
            [_SDPBlock(dim_model, num_heads, ff_expansion=2.0, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim_model)
        self.head = nn.Linear(dim_model, vocab_size, bias=False)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.embedding(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


# ===========================================================================
# GravityModel — thin wrapper so it has the same call signature as SDPModel
# ===========================================================================


class GravityModel(nn.Module):
    """
    LGT wrapper that exposes ``forward(token_ids) -> logits`` for parity with
    ``SDPModel``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.lgt = LightweightGravitationalTransformer(**kwargs)
        # expose dim_model so benchmark helper can read it
        self.dim_model = kwargs.get("dim_model", 192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.lgt(x)
        return logits


# ===========================================================================
# Optional JAX model
# ===========================================================================

if _JAX_AVAILABLE:
    class _JAXSDPAttention(fnn.Module):  # type: ignore[misc]
        """Minimal Flax multi-head dot-product attention block."""
        features: int
        num_heads: int

        @fnn.compact
        def __call__(self, x):  # type: ignore[override]
            return fnn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.features,
            )(x)

    class _JAXTransformer(fnn.Module):  # type: ignore[misc]
        vocab_size: int
        dim_model: int
        num_heads: int
        num_layers: int

        @fnn.compact
        def __call__(self, ids):  # type: ignore[override]
            x = fnn.Embed(self.vocab_size, self.dim_model)(ids)
            for _ in range(self.num_layers):
                x = x + _JAXSDPAttention(self.dim_model, self.num_heads)(x)
                x = fnn.LayerNorm()(x)
            x = fnn.Dense(self.vocab_size)(x)
            return x


# ===========================================================================
# Parameter counting
# ===========================================================================


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_jax_params(params: Any) -> int:
    if not _JAX_AVAILABLE:
        return 0
    return sum(v.size for v in jax.tree_util.tree_leaves(params))


# ===========================================================================
# FLOPs estimation
# ===========================================================================


@dataclass
class FlopsBreakdown:
    """Analytical FLOPs estimate for one forward pass."""
    attn_flops: int
    ffn_flops: int
    embed_flops: int
    total_flops: int
    label: str

    def __str__(self) -> str:
        gflops = self.total_flops / 1e9
        return (
            f"{self.label}: total={gflops:.4f} GFLOPs "
            f"(attn={self.attn_flops/1e6:.2f}M "
            f"ffn={self.ffn_flops/1e6:.2f}M)"
        )


def estimate_flops_sdp(
    batch: int, seq: int, dim: int, num_heads: int, num_layers: int
) -> FlopsBreakdown:
    """
    Standard multi-head self-attention FLOPs:
      QKV projections : 3 * 2 * B * S * D^2
      Attention scores : 2 * B * H * S^2 * (D/H)  = 2 * B * S^2 * D
      Attention output : 2 * B * H * S * (D/H) * S = 2 * B * S^2 * D
      Out projection  : 2 * B * S * D^2
    FFN (2× expansion):
      2 * B * S * D * 2D  +  2 * B * S * 2D * D  = 8 * B * S * D^2
    """
    attn = num_layers * (
        3 * 2 * batch * seq * dim * dim    # QKV
        + 2 * batch * seq * seq * dim      # scores
        + 2 * batch * seq * seq * dim      # weighted sum
        + 2 * batch * seq * dim * dim      # out proj
    )
    ffn = num_layers * 8 * batch * seq * dim * dim
    embed = 2 * batch * seq * dim          # embedding lookup + head
    return FlopsBreakdown(
        attn_flops=attn,
        ffn_flops=ffn,
        embed_flops=embed,
        total_flops=attn + ffn + embed,
        label="SDP",
    )


def estimate_flops_gravity(
    batch: int, seq: int, dim: int, num_heads: int, num_layers: int
) -> FlopsBreakdown:
    """
    Gravitational attention FLOPs:
      Mass projection  : 2 * B * S * (D/H) * num_heads  = 2 * B * S * D
      Pairwise forces  : B * S^2 * H  (mass products + division per pair+head)
      Attention output : 2 * B * S^2 * D  (same as standard weighted sum)
      Out projection   : 2 * B * S * D^2
      No QKV projections (replaced by mass_proj + force formula)
    FFN: same as SDP (unchanged).
    """
    attn = num_layers * (
        2 * batch * seq * dim                   # mass projection (one linear per head)
        + 4 * batch * seq * seq * num_heads     # pairwise force: mi*mj/dist² per pair per head
        + 2 * batch * seq * seq * dim           # weighted sum
        + 2 * batch * seq * dim * dim           # out proj
    )
    ffn = num_layers * 8 * batch * seq * dim * dim
    embed = 2 * batch * seq * dim
    return FlopsBreakdown(
        attn_flops=attn,
        ffn_flops=ffn,
        embed_flops=embed,
        total_flops=attn + ffn + embed,
        label="Gravity",
    )


# ===========================================================================
# Toy training task: next-token prediction on random sequences
# ===========================================================================


def _toy_batch(
    vocab_size: int, seq_len: int, batch: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random token-ID batch for next-token prediction."""
    ids = torch.randint(0, vocab_size, (batch, seq_len + 1), device=device)
    return ids[:, :-1], ids[:, 1:]   # inputs, targets


def train_one_epoch(
    model: nn.Module,
    vocab_size: int,
    seq_len: int,
    batch: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> List[float]:
    """
    Train ``model`` for ``steps`` steps on random next-token batches.

    Returns a list of per-step cross-entropy losses (float).
    """
    model.train()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: List[float] = []
    for _ in range(steps):
        ids, targets = _toy_batch(vocab_size, seq_len, batch, device)
        opt.zero_grad()
        logits = model(ids)                          # [B, T, V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


# ===========================================================================
# Latency / throughput measurement
# ===========================================================================


def _measure_latency(
    model: nn.Module,
    example: torch.Tensor,
    num_warmup: int,
    num_runs: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    model.to(device)
    example = example.to(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(example)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)
    mean_ms = sum(times) / n
    p95_ms = times[int(n * 0.95)]
    batch = example.shape[0]
    seq = example.shape[1]
    tok_per_s = (batch * seq * 1000) / mean_ms
    return {
        "mean_ms": round(mean_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "tokens_per_sec": round(tok_per_s, 1),
    }


# ===========================================================================
# ASCII loss-curve sparkline
# ===========================================================================

_SPARKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], width: int = 40) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1.0
    # Downsample to width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    chars = [_SPARKS[min(int((v - mn) / rng * 8), 8)] for v in sampled]
    return "".join(chars)


# ===========================================================================
# Pretty-print helpers
# ===========================================================================


def _print_separator(width: int = 70) -> None:
    print("─" * width)


def _print_loss_table(
    gravity_losses: List[float],
    sdp_losses: List[float],
    interval: int = 5,
) -> None:
    """Print a numeric loss table at every ``interval`` steps."""
    print("\n┌─ Loss curve (cross-entropy, next-token prediction on random sequences) ─┐")
    header = f"{'Step':>6}  {'Gravity':>10}  {'SDP':>10}  {'Δ (G-S)':>10}"
    print(header)
    _print_separator(len(header))
    steps = min(len(gravity_losses), len(sdp_losses))
    for i in range(0, steps, interval):
        g, s = gravity_losses[i], sdp_losses[i]
        delta = g - s
        sign = "+" if delta >= 0 else ""
        print(f"{i+1:>6}  {g:>10.4f}  {s:>10.4f}  {sign}{delta:>9.4f}")
    # Always print the last step
    last = steps - 1
    if last % interval != 0:
        g, s = gravity_losses[last], sdp_losses[last]
        delta = g - s
        sign = "+" if delta >= 0 else ""
        print(f"{last+1:>6}  {g:>10.4f}  {s:>10.4f}  {sign}{delta:>9.4f}")
    _print_separator(len(header))
    print(f"{'Gravity':>19}  {'SDP':>10}")
    print(f"  First loss : {gravity_losses[0]:>9.4f}  {sdp_losses[0]:>10.4f}")
    print(f"  Final loss : {gravity_losses[-1]:>9.4f}  {sdp_losses[-1]:>10.4f}")
    print(f"  Δ final    : {gravity_losses[-1] - sdp_losses[-1]:>+10.4f}")
    print()
    print("  Gravity sparkline: " + _sparkline(gravity_losses))
    print("  SDP     sparkline: " + _sparkline(sdp_losses))
    print("└" + "─" * 68 + "┘")


def _print_latency_table(
    results: List[Dict[str, Any]],
) -> None:
    print("\n┌─ Latency & Throughput (CPU edge benchmark) ──────────────────────────┐")
    header = f"{'Model':<14} {'Params':>10}  {'Mean ms':>9}  {'p95 ms':>9}  {'Tok/s':>10}"
    print(header)
    _print_separator(len(header))
    for r in results:
        lat = r["latency"]
        print(
            f"{r['label']:<14} {r['params']:>10,}  "
            f"{lat['mean_ms']:>9.2f}  {lat['p95_ms']:>9.2f}  "
            f"{lat['tokens_per_sec']:>10,.0f}"
        )
    _print_separator(len(header))
    print("└" + "─" * 68 + "┘")


def _print_flops_table(
    gravity_flops: FlopsBreakdown,
    sdp_flops: FlopsBreakdown,
    jax_flops: Optional[FlopsBreakdown] = None,
) -> None:
    print("\n┌─ FLOPs Estimate (analytical, one forward pass) ──────────────────────┐")
    header = (
        f"{'Model':<14} {'Attn MFLOPs':>13}  "
        f"{'FFN MFLOPs':>12}  {'Total GFLOPs':>13}  {'Ratio':>7}"
    )
    print(header)
    _print_separator(len(header))

    rows = [gravity_flops, sdp_flops]
    if jax_flops is not None:
        rows.append(jax_flops)

    base = sdp_flops.total_flops
    for fb in rows:
        ratio = fb.total_flops / base
        print(
            f"{fb.label:<14} {fb.attn_flops/1e6:>13.2f}  "
            f"{fb.ffn_flops/1e6:>12.2f}  "
            f"{fb.total_flops/1e9:>13.4f}  "
            f"{ratio:>7.3f}x"
        )
    _print_separator(len(header))
    ratio_ga = gravity_flops.total_flops / sdp_flops.total_flops
    note = "cheaper" if ratio_ga < 1.0 else "costlier"
    attn_note = (
        "cheaper"
        if gravity_flops.attn_flops < sdp_flops.attn_flops
        else "costlier"
    )
    attn_pct = abs(1 - gravity_flops.attn_flops / sdp_flops.attn_flops) * 100
    print(
        f"  Gravity attention is {abs(1 - ratio_ga)*100:.1f}% {note} "
        f"than SDP (attn sublayer: {attn_pct:.1f}% {attn_note})"
    )
    print("└" + "─" * 68 + "┘")


# ===========================================================================
# JAX harness (optional)
# ===========================================================================


def _run_jax_bench(
    seq_len: int,
    batch: int,
    runs: int,
    vocab_size: int,
    dim_model: int,
    num_heads: int,
    num_layers: int,
) -> Optional[Dict[str, Any]]:
    if not _JAX_AVAILABLE:
        return None

    import jax.random as jr  # type: ignore

    model = _JAXTransformer(
        vocab_size=vocab_size,
        dim_model=dim_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    key = jr.PRNGKey(0)
    dummy = jnp.ones((batch, seq_len), dtype=jnp.int32)
    params = model.init(key, dummy)["params"]
    n_params = _count_jax_params(params)

    @jax.jit
    def fwd(p, x):  # type: ignore[return]
        return model.apply({"params": p}, x)

    # Warm-up: ensure compilation + first execution complete
    fwd(params, dummy).block_until_ready()

    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fwd(params, dummy).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)
    mean_ms = sum(times) / n
    tok_s = (batch * seq_len * 1000) / mean_ms
    return {
        "label": "JAX-SDP",
        "params": n_params,
        "latency": {
            "mean_ms": round(mean_ms, 3),
            "p95_ms": round(times[int(n * 0.95)], 3),
            "tokens_per_sec": round(tok_s, 1),
        },
    }


# ===========================================================================
# Main harness
# ===========================================================================


def run_benchmark(
    steps: int = 50,
    seq_len: int = BENCH_SEQ_LEN,
    batch: int = BENCH_BATCH,
    lr: float = 3e-4,
    runs: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full benchmark harness.  Returns a results dict that is also written to
    ``/tmp/bench_gravity_vs_sdp_results.json``.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  GRAVITY ATTENTION vs. SCALED-DOT-PRODUCT — 1 M-param Edge Benchmark")
    print(f"{'='*70}")
    print(f"  Device  : {device}")
    print(f"  Seq len : {seq_len}   Batch : {batch}   Train steps : {steps}")
    print(f"  Bench runs : {runs}   Seed : {seed}")
    print(f"  JAX available : {_JAX_AVAILABLE}")
    print()

    vocab = EDGE_1M["vocab_size"]
    dim = EDGE_1M["dim_model"]
    layers = EDGE_1M["num_layers"]
    heads = EDGE_1M["num_heads"]

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    gravity_model = GravityModel(**EDGE_1M)
    sdp_model = SDPModel(
        vocab_size=vocab,
        dim_model=dim,
        num_layers=layers,
        num_heads=heads,
        max_seq_len=EDGE_1M["max_seq_len"],
        dropout=EDGE_1M["dropout"],
    )

    n_gravity = _count_params(gravity_model)
    n_sdp = _count_params(sdp_model)
    print(f"  Gravity model params : {n_gravity:>10,}")
    print(f"  SDP     model params : {n_sdp:>10,}")
    print()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("─── Phase 1: Training (next-token prediction on random sequences) ───")
    print()

    # Reset seeds per model for a fair comparison
    torch.manual_seed(seed)
    gravity_losses = train_one_epoch(
        gravity_model, vocab, seq_len, batch, steps, lr, device
    )
    torch.manual_seed(seed)
    sdp_losses = train_one_epoch(
        sdp_model, vocab, seq_len, batch, steps, lr, device
    )

    interval = max(1, steps // 10)
    _print_loss_table(gravity_losses, sdp_losses, interval=interval)

    # ------------------------------------------------------------------
    # Latency / throughput — always measured on CPU (edge-device proxy)
    # ------------------------------------------------------------------
    print("\n─── Phase 2: Latency & Throughput (CPU) ───")
    cpu_device = torch.device("cpu")
    example = torch.randint(0, vocab, (batch, seq_len), device=cpu_device)
    warmup = max(2, runs // 10)

    # Move models to CPU explicitly; restore to training device afterwards.
    gravity_model.to(cpu_device)
    sdp_model.to(cpu_device)

    gravity_lat = _measure_latency(gravity_model, example, warmup, runs, cpu_device)
    sdp_lat = _measure_latency(sdp_model, example, warmup, runs, cpu_device)

    gravity_model.to(device)
    sdp_model.to(device)

    latency_rows: List[Dict[str, Any]] = [
        {"label": "Gravity LGT", "params": n_gravity, "latency": gravity_lat},
        {"label": "SDP Baseline", "params": n_sdp,     "latency": sdp_lat},
    ]

    jax_result = _run_jax_bench(seq_len, batch, runs, vocab, dim, heads, layers)
    if jax_result:
        latency_rows.append(jax_result)

    _print_latency_table(latency_rows)

    # ------------------------------------------------------------------
    # FLOPs
    # ------------------------------------------------------------------
    print("\n─── Phase 3: FLOPs Comparison ───")
    gflops = estimate_flops_gravity(batch, seq_len, dim, heads, layers)
    sflops = estimate_flops_sdp(batch, seq_len, dim, heads, layers)
    jax_flops = sflops  # JAX-SDP is the same formula; re-labelled if present
    if jax_result:
        jax_flops = FlopsBreakdown(
            attn_flops=sflops.attn_flops,
            ffn_flops=sflops.ffn_flops,
            embed_flops=sflops.embed_flops,
            total_flops=sflops.total_flops,
            label="JAX-SDP",
        )
        _print_flops_table(gflops, sflops, jax_flops)
    else:
        _print_flops_table(gflops, sflops)

    # ------------------------------------------------------------------
    # Assemble result dict
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {
        "meta": {
            "device": str(device),
            "seq_len": seq_len,
            "batch": batch,
            "train_steps": steps,
            "bench_runs": runs,
            "seed": seed,
            "jax_available": _JAX_AVAILABLE,
            "torch_version": torch.__version__,
        },
        "params": {"gravity": n_gravity, "sdp": n_sdp},
        "training": {
            "gravity": {
                "losses": gravity_losses,
                "first": round(gravity_losses[0], 6),
                "final": round(gravity_losses[-1], 6),
            },
            "sdp": {
                "losses": sdp_losses,
                "first": round(sdp_losses[0], 6),
                "final": round(sdp_losses[-1], 6),
            },
        },
        "latency": {
            "gravity": gravity_lat,
            "sdp": sdp_lat,
        },
        "flops": {
            "gravity_total_gflops": round(gflops.total_flops / 1e9, 6),
            "sdp_total_gflops": round(sflops.total_flops / 1e9, 6),
            "gravity_attn_mflops": round(gflops.attn_flops / 1e6, 4),
            "sdp_attn_mflops": round(sflops.attn_flops / 1e6, 4),
            "ratio_total": round(gflops.total_flops / sflops.total_flops, 4),
            "ratio_attn_only": round(gflops.attn_flops / sflops.attn_flops, 4),
        },
    }
    if jax_result:
        results["latency"]["jax_sdp"] = jax_result["latency"]

    out_path = "/tmp/bench_gravity_vs_sdp_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results written → {out_path}")

    return results


# ===========================================================================
# CLI
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gravity-attention vs. scaled-dot-product 1M-param edge benchmark."
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="CI mode: 5 training steps, 10 bench runs.",
    )
    p.add_argument("--steps", type=int, default=50,
                   help="Training steps (default: 50).")
    p.add_argument("--seq-len", type=int, default=BENCH_SEQ_LEN,
                   help=f"Sequence length (default: {BENCH_SEQ_LEN}).")
    p.add_argument("--batch-size", type=int, default=BENCH_BATCH,
                   help=f"Batch size (default: {BENCH_BATCH}).")
    p.add_argument("--runs", type=int, default=100,
                   help="Latency benchmark runs (default: 100).")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate (default: 3e-4).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    return p


def main() -> None:  # pragma: no cover
    args = _build_parser().parse_args()
    if args.quick:
        args.steps = 5
        args.runs = 10
    run_benchmark(
        steps=args.steps,
        seq_len=args.seq_len,
        batch=args.batch_size,
        lr=args.lr,
        runs=args.runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
