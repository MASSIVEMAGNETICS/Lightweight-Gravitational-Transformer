# Lightweight Gravitational Transformer (LGT)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer/actions/workflows/ci.yml)

A **physics-aware transformer architecture** that replaces standard query-key-value attention with Newton's law of gravitation, producing a minimal yet powerful model optimised for resource-constrained environments, edge deployment, and VictorOS cognitive-runtime integration.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Configuration Reference](#configuration-reference)
- [Training](#training)
- [Edge Export](#edge-export)
- [VictorOS Integration](#victoros-integration)
- [Tri-Model Architecture](#tri-model-architecture)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Lightweight Gravitational Transformer** (LGT) computes attention weights from *gravitational forces* between tokens rather than from softmax-scaled dot products. Each token is assigned a learnable mass; attention from token *i* to token *j* is proportional to the gravitational force:

```
F_ij = G · m_i · m_j / (dist(p_i, p_j)² + ε)
```

This formulation:

- Naturally encodes **distance-sensitive attention** via curved positional manifolds.
- Provides **physical interpretability** — you can inspect masses and forces directly.
- Includes built-in **stability guarantees** (Hawking regularisation, Bekenstein entropy penalty, ContainmentProtocol).
- Achieves competitive quality at **≤150 K parameters** on constrained hardware.

---

## Key Features

| Feature | Description |
|---|---|
| **Gravitational Attention** | Newton-law force-based attention with per-head learnable `G` |
| **Curved / Fractal Positions** | Two position-encoding strategies: curved manifold or fractal power-law |
| **ContainmentProtocol** | Runtime safety guard: gradient clipping, force dampening, entropy regularisation |
| **MetaCurvatureScheduler** | Self-evolving positional geometry driven by validation loss |
| **Mirror Layer** | Real-time introspection hook streaming diagnostics to the VictorOS Cortex |
| **Ledger** | Append-only JSONL audit trail for every inference and training event |
| **Tri-Model Fusion** | World / Self / Environment cross-gravitational architecture |
| **Edge Export** | TorchScript tracing + INT8 / FP16 quantisation with four preset configs |
| **VictorOS Module** | `@victoros_module` decorator for first-class cognitive-agent packaging |

---

## Architecture

```
Input tokens / embeddings
        │
        ▼
 ┌──────────────────────┐
 │  Token Embedding     │  (optional, for discrete vocabularies)
 └──────────┬───────────┘
            │
 ┌──────────▼───────────┐
 │  Position Embedding  │  CurvedPositionEmbedding  OR
 │                      │  FractalPositionEmbedding
 └──────────┬───────────┘
            │  positions [seq, dim_pos]
   ┌────────▼─────────────────────────────────────┐
   │  LightweightGravitationalBlock  × num_layers  │
   │                                               │
   │   ┌─────────────────────────────────────┐     │
   │   │  MultiHeadGravitationalAttention    │     │
   │   │   • per-head learnable G            │     │
   │   │   • mass_proj: token → scalar mass  │     │
   │   │   • F_ij = G·m_i·m_j / dist²       │     │
   │   │   • Hawking clamp (max_force)       │     │
   │   └───────────────┬─────────────────────┘     │
   │                   │ residual + LayerNorm        │
   │   ┌───────────────▼─────────────────────┐     │
   │   │  Lightweight FFN (2× expansion)     │     │
   │   └─────────────────────────────────────┘     │
   └────────────────────┬─────────────────────────┘
                        │
              LayerNorm + (optional) LM Head
                        │
                     Output
```

### Gravitational Attention in Detail

```python
# 1. Each token projects to a scalar mass
masses = softplus(mass_proj(x))          # always positive

# 2. Pairwise distances from curved positions
dist_sq = ||p_i - p_j||² + event_horizon
if curvature != 0:
    dist_sq *= (1 + curvature * cos(||p||))  # space curvature

# 3. Gravitational force matrix
F_ij = |G| * m_i * m_j / dist_sq

# 4. Hawking regularisation (prevent attention collapse)
F_ij = clamp(F_ij, max=max_force)

# 5. Softmax → attention weights
attn = softmax(F_ij, dim=-1)
```

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0

### From Source (recommended)

```bash
# Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer.git
cd Lightweight-Gravitational-Transformer

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements_lgt.txt

# Optional: install as an editable package
pip install -e .
```

### Using pip (once published)

```bash
pip install lightweight-gravitational-transformer
```

### Verify Installation

```python
import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

model = LightweightGravitationalTransformer(vocab_size=1000, dim_model=64)
x = torch.randint(0, 1000, (1, 16))
output, _ = model(x)
print(output.shape)   # torch.Size([1, 16, 64])
print("LGT installed correctly ✓")
```

---

## Quick Start

### Minimal Inference

```python
import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

# Build a small model (no vocabulary — accepts continuous embeddings)
model = LightweightGravitationalTransformer(
    dim_model=128,
    dim_position=64,
    num_layers=4,
    num_heads=4,
)

# Continuous embedding input [batch, seq_len, dim_model]
x = torch.randn(2, 32, 128)
output, diagnostics = model(x, return_diagnostics=True)

print(output.shape)                    # [2, 32, 128]
print(diagnostics["curvature"])        # 0.15
```

### Language-Model Mode

```python
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

model = LightweightGravitationalTransformer(
    vocab_size=32000,
    dim_model=256,
    num_layers=6,
    num_heads=8,
    max_seq_len=512,
    tie_weights=True,          # tie input embedding ↔ output projection
)

# Token IDs [batch, seq_len]
token_ids = torch.randint(0, 32000, (2, 64))
logits, _ = model(token_ids)
print(logits.shape)                    # [2, 64, 32000]
```

### Fractal Position Embeddings

```python
model = LightweightGravitationalTransformer(
    dim_model=128,
    use_fractal_positions=True,
    fractal_dim=1.5,           # Hausdorff-like dimension
)
```

---

## Core Modules

### `gravitational_attention.py`

#### `GravitationalAttentionHead`

Single attention head using gravitational force computation.

```python
from gravitational_attention import GravitationalAttentionHead

head = GravitationalAttentionHead(
    head_dim=32,
    gravitational_constant=1.0,  # initial G (learnable)
    event_horizon=1e-6,          # minimum distance² (prevents division by zero)
    max_force=50.0,              # Hawking regularisation cap (None to disable)
    curvature=0.15,              # spacetime curvature applied to distances
)

x = torch.randn(2, 16, 32)      # [batch, seq, head_dim]
out, masses = head(x)
print(masses.shape)              # [batch, seq]  — per-token masses
```

#### `MultiHeadGravitationalAttention`

Drop-in multi-head extension with independent per-head `G` values.

```python
from gravitational_attention import MultiHeadGravitationalAttention

attn = MultiHeadGravitationalAttention(
    dim_model=128,
    num_heads=4,
    different_G_per_head=True,  # each head learns its own gravitational constant
)

x = torch.randn(2, 16, 128)
out = attn(x)                   # [batch, seq, dim_model]

# Diagnostic introspection
diag = attn.get_attention_diagnostics(x)
print(diag["head_0"])           # {"mean_mass", "mean_force", "G", "curvature"}
```

---

### `fractal_position_embedding.py`

#### `FractalPositionEmbedding`

Multi-scale sinusoidal embedding with power-law frequency spacing.

```python
from fractal_position_embedding import FractalPositionEmbedding

embed = FractalPositionEmbedding(
    max_seq_len=512,
    dim_position=64,
    fractal_dim=1.5,     # > 1 compresses high-frequency scales
    num_scales=4,
    learnable_residual=True,
)

positions = embed(seq_len=32)   # [32, 64]
```

---

### `lightweight_gravitational_transformer.py`

#### `LightweightGravitationalTransformer`

Full model stack. Key constructor parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int \| None` | `None` | Vocabulary size; `None` for continuous input |
| `dim_model` | `int` | `128` | Model / embedding dimension |
| `dim_position` | `int` | `64` | Position vector dimension |
| `num_layers` | `int` | `4` | Number of gravitational blocks |
| `num_heads` | `int` | `4` | Attention heads per block |
| `max_seq_len` | `int` | `512` | Maximum sequence length |
| `curvature` | `float` | `0.15` | Spacetime curvature for positional embeddings |
| `gravitational_constant` | `float` | `1.0` | Base G (decays as `G × 0.9^layer`) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `tie_weights` | `bool` | `False` | Tie embedding ↔ output projection |
| `use_fractal_positions` | `bool` | `False` | Use fractal instead of curved positions |
| `fractal_dim` | `float` | `1.5` | Hausdorff dimension for fractal positions |

**Forward signature:**
```python
output, diagnostics = model(
    x,                                 # [batch, seq, dim] or token IDs
    positions=None,                    # override position vectors
    return_diagnostics=False,          # enable introspection
    mirror_layer_callback=None,        # MirrorLayer callback
)
```

---

### `victorcos_module.py`

#### `Ledger`

Append-only structured event log with optional JSONL persistence.

```python
from victorcos_module import Ledger

ledger = Ledger(
    agent_id="my_agent",
    persist_path="logs/agent.jsonl",  # None for memory-only
    max_memory_entries=1000,
)

ledger.log("inference", {"seq_len": 32, "output_mean": 0.01})
ledger.log("checkpoint", {"path": "ckpt.pt"})

entries = ledger.entries(event_filter="inference")
ledger.flush()                        # write to disk
```

#### `MirrorLayer`

Real-time stability monitor that hooks into the model's forward pass.

```python
from victorcos_module import Ledger, MirrorLayer

ledger = Ledger(agent_id="mirror")
mirror = MirrorLayer(
    ledger=ledger,
    max_force_threshold=40.0,
    stability_window=20,
    correction_callback=lambda layer, correction: print(f"[{layer}] {correction}"),
)

# Pass as callback to model.forward()
output, _ = model(x, return_diagnostics=True, mirror_layer_callback=mirror)
print(mirror.stability_score())       # float in [0, 1]
```

#### `@victoros_module` Decorator

```python
from victorcos_module import victoros_module, VictorOSBaseModule

@victoros_module(
    name="my_lgt_agent",
    version="1.0.0",
    containment_native=True,
    description="Custom LGT cognitive module.",
)
class MyAgent(VictorOSBaseModule):
    def __init__(self, model):
        self.model = model

    def process(self, x):
        output, diag = self.model(x, return_diagnostics=True,
                                   mirror_layer_callback=self.mirror_layer)
        self.ledger.log("inference", {"stability": self.mirror_layer.stability_score()})
        return output
```

#### `LGTVictorOSModule`

Pre-built VictorOS module wrapping any `LightweightGravitationalTransformer`.

```python
from victorcos_module import LGTVictorOSModule

module = LGTVictorOSModule(
    model=model,
    agent_id="lgt_core",
    persist_path="ledger.jsonl",
    max_force_threshold=40.0,
)

result = module.process(x)
# result = {"output": tensor, "diagnostics": {...}, "stability": float}

# Self-evolution proposal
proposal = module.propose_architecture_change(
    current_config={"num_layers": 4, "curvature": 0.15},
    stability_threshold=0.95,
)
```

---

### `training.py`

#### `ContainmentProtocol`

Per-step safety guard that wraps the training loop.

```python
from training import ContainmentConfig, ContainmentProtocol

config = ContainmentConfig(
    max_grad_norm=1.0,           # gradient clipping threshold
    max_attention_force=40.0,    # force dampening threshold
    bekenstein_lambda=1e-4,      # entropy regularisation weight
    min_loss=1e-8,               # collapse detection
    max_loss=1e4,                # divergence detection
)

protocol = ContainmentProtocol(config=config, model=model, ledger=ledger)

# After loss.backward(), before optimizer.step():
summary = protocol.step(loss, diagnostics)
if summary["stopped"]:
    print("Training halted by ContainmentProtocol")
if summary["proposal"]:
    print("Architecture proposal:", summary["proposal"])
```

#### `TrainingLoop`

Full training orchestrator with physics-aware constraints.

```python
from training import TrainingLoop, TrainingConfig, ContainmentConfig
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    loss_fn=torch.nn.CrossEntropyLoss(),
    config=TrainingConfig(
        max_steps=10_000,
        eval_every=500,
        use_bekenstein_penalty=True,
        use_meta_curvature=True,
    ),
    containment_config=ContainmentConfig(),
    ledger=ledger,
)

summary = loop.fit(train_iter, val_iter=val_iter, on_proposal=print)
print(summary)  # {"steps": ..., "final_loss": ..., "proposals": [...]}
```

---

### `tri_model.py`

#### `TriModelTransformer`

Three-stream cognitive architecture for world / self / environment fusion.

```python
from tri_model import TriModelTransformer

tri = TriModelTransformer(
    dim_model=128,
    num_layers=4,
    num_heads=4,
    vocab_size=32000,            # optional; set if inputs are token IDs
    output_dim=128,
)

world = torch.randn(2, 32, 128)
self_ = torch.randn(2, 16, 128)
env   = torch.randn(2, 8,  128)

output, diagnostics = tri(world, self_, env, return_diagnostics=True)
print(output.shape)              # [2, 32, 128]
```

---

### `export_edge_model.py`

#### Export Presets

| Preset | `dim_model` | Layers | Heads | ~Params | ~FP32 Size |
|---|---|---|---|---|---|
| `edge_150k` | 64 | 2 | 2 | ~150 K | <1 MB |
| `meta_probe` | 128 | 4 | 4 | ~600 K | ~2.3 MB |
| `victorcos` | 192 | 5 | 6 | ~1.4 M | ~5.3 MB |
| `fractal_res` | 256 | 6 | 8 | ~2.1 M | ~8.0 MB |

```python
from export_edge_model import export_edge_model

paths = export_edge_model(
    config_name="edge_150k",
    vocab_size=32000,
    quantize="int8",             # "none" | "int8" | "float16"
    output_dir="exported_models",
    use_fractal_positions=False,
)
print(paths["checkpoint"])       # exported_models/lgt_edge_150k_int8.pt
```

**CLI:**
```bash
python export_edge_model.py \
  --config edge_150k \
  --quantize int8 \
  --output-dir exported_models \
  --vocab-size 32000
```

---

## Configuration Reference

### `ContainmentConfig`

```python
@dataclass
class ContainmentConfig:
    max_grad_norm: float = 1.0
    max_attention_force: float = 40.0
    bekenstein_lambda: float = 1e-4
    min_loss: float = 1e-8
    max_loss: float = 1e4
    stability_ema_alpha: float = 0.05
    enable_architecture_proposals: bool = True
    stability_proposal_threshold: float = 0.95
    proposal_min_interval: int = 100
```

### `TrainingConfig`

```python
@dataclass
class TrainingConfig:
    max_steps: int = 10_000
    eval_every: int = 500
    log_every: int = 50
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    use_bekenstein_penalty: bool = True
    use_meta_curvature: bool = True
    meta_curvature_lr: float = 0.01
    grad_accumulation_steps: int = 1
```

---

## Training

### Basic Training Loop

```python
import torch
import torch.nn as nn
from lightweight_gravitational_transformer import LightweightGravitationalTransformer
from training import TrainingLoop, TrainingConfig, ContainmentConfig
from victorcos_module import Ledger

# Model
model = LightweightGravitationalTransformer(
    vocab_size=1000,
    dim_model=128,
    num_layers=4,
    num_heads=4,
)

# Ledger for audit trail
ledger = Ledger(agent_id="train_run_001", persist_path="logs/train.jsonl")

# Optimiser + loss
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

# Training loop
loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    loss_fn=lambda logits, targets: loss_fn(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    ),
    config=TrainingConfig(max_steps=5000, eval_every=250),
    containment_config=ContainmentConfig(max_grad_norm=1.0),
    ledger=ledger,
)

# Synthetic data iterator
def data_iter(vocab_size=1000, seq_len=32, batch_size=8):
    while True:
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        yield x, y

summary = loop.fit(data_iter(), on_proposal=lambda p: print("Proposal:", p))
print(f"Finished in {summary['steps']} steps, final loss = {summary['final_loss']:.4f}")
ledger.flush()
```

### Training with Mirror Layer

```python
from victorcos_module import MirrorLayer

mirror = MirrorLayer(ledger=ledger, max_force_threshold=35.0)

# Single training step with Mirror Layer diagnostics
result = loop.train_step(
    batch=(x_batch, y_batch),
    return_diagnostics=True,    # enables mirror_layer_callback
)
print(f"Stability: {result['stability']:.3f}")
```

---

## Edge Export

```bash
# Export smallest preset with INT8 quantisation
python export_edge_model.py --config edge_150k --quantize int8

# Export for VictorOS integration (FP16)
python export_edge_model.py --config victorcos --quantize float16

# Export full-size model without quantisation
python export_edge_model.py --config fractal_res --quantize none
```

### Load Exported Checkpoint

```python
import torch

state = torch.load("exported_models/lgt_edge_150k_int8.pt", weights_only=False)
print(state["metadata"])         # config, vocab_size, n_params, …
```

---

## VictorOS Integration

LGT is designed as a first-class cognitive module for the VictorOS runtime:

```
VictorOS Cortex
    │
    ├── @victoros_module ──► LGTVictorOSModule
    │        │
    │        ├── Ledger  (append-only JSONL audit trail)
    │        ├── MirrorLayer  (real-time stability monitoring)
    │        └── LightweightGravitationalTransformer
    │
    └── Architecture Proposals ──► Cortex applies structural changes
```

### Registering a Custom Module

```python
@victoros_module(
    name="custom_lgt",
    version="1.0.0",
    requirements=["torch>=2.0.0"],
    containment_native=True,
    description="Custom physics-aware cognitive module.",
)
class CustomLGTModule(VictorOSBaseModule):
    def __init__(self):
        self.model = LightweightGravitationalTransformer(dim_model=128)

    def process(self, x):
        output, _ = self.model(
            x,
            return_diagnostics=True,
            mirror_layer_callback=self.mirror_layer,
        )
        self.ledger.log("inference", {"output_norm": float(output.norm())})
        return output
```

---

## Tri-Model Architecture

The Tri-Model Transformer implements a three-stream cognitive architecture where:

- **WorldModel** (curvature=0.25, G=1.0) — external semantic context
- **SelfModel** (curvature=0.15, G=0.8) — agent internal state
- **EnvironmentModel** (curvature=0.10, G=1.2) — interaction urgency

The three streams are fused via `CrossGravitationalFusion`, where each stream's mean representation acts as a gravitational mass that exerts influence on the other two.

```python
from tri_model import TriModelTransformer

model = TriModelTransformer(
    dim_model=128,
    num_layers=4,
    num_heads=4,
    vocab_size=32000,
)

world_tokens = torch.randint(0, 32000, (1, 32))
self_tokens  = torch.randint(0, 32000, (1, 16))
env_tokens   = torch.randint(0, 32000, (1, 8))

output, diagnostics = model(world_tokens, self_tokens, env_tokens)

# VictorOS causal trace
snapshot = model.get_tri_snapshot(world_tokens, self_tokens, env_tokens)
```

---

## Examples

See the [`examples/`](examples/) directory for runnable scripts:

| Script | Description |
|---|---|
| [`examples/basic_inference.py`](examples/basic_inference.py) | Minimal forward pass with continuous embeddings |
| [`examples/language_model.py`](examples/language_model.py) | Token-ID language model with training loop |
| [`examples/victorcos_integration.py`](examples/victorcos_integration.py) | VictorOS module, Ledger, and Mirror Layer |
| [`examples/edge_export.py`](examples/edge_export.py) | Export model for edge deployment |
| [`examples/tri_model_fusion.py`](examples/tri_model_fusion.py) | Tri-model world/self/environment fusion |

---

## Benchmarks

Run the benchmark suite:

```bash
python benchmarks/benchmark_lgt.py
```

This measures:
- Inference latency and throughput across all four presets
- Memory footprint (FP32 / FP16 / INT8)
- Forward-pass time per sequence length

---

## Running Tests

```bash
# Install test dependencies (pytest is sufficient)
pip install pytest

# Run the full test suite
pytest tests/ -v

# Run a specific test class
pytest tests/test_lgt.py::TestGravitationalAttentionHead -v

# Run with coverage (requires pytest-cov)
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Project Structure

```
Lightweight-Gravitational-Transformer/
├── gravitational_attention.py          # Core gravitational attention mechanism
├── fractal_position_embedding.py       # Multi-scale fractal position encoding
├── lightweight_gravitational_transformer.py  # Main transformer stack
├── victorcos_module.py                 # VictorOS Ledger, MirrorLayer, @victoros_module
├── training.py                         # ContainmentProtocol, MetaCurvature, TrainingLoop
├── tri_model.py                        # Tri-model world/self/env fusion
├── export_edge_model.py                # Edge quantisation and TorchScript export
├── requirements_lgt.txt                # Python dependencies
├── pyproject.toml                      # Package metadata and build config
├── examples/                           # Runnable usage examples
│   ├── basic_inference.py
│   ├── language_model.py
│   ├── victorcos_integration.py
│   ├── edge_export.py
│   └── tri_model_fusion.py
├── tests/
│   └── test_lgt.py                     # 60+ pytest test cases
├── benchmarks/
│   └── benchmark_lgt.py               # Performance benchmarking
└── docs/
    ├── installation.md                 # Detailed installation guide
    ├── user_guide.md                   # In-depth user guide
    ├── api.md                          # Full API reference
    └── architecture.md                # Architecture deep-dive
```

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and the pull-request process.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use LGT in academic work, please cite:

```bibtex
@software{lgt2024,
  title  = {Lightweight Gravitational Transformer},
  author = {MASSIVEMAGNETICS},
  year   = {2024},
  url    = {https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer},
}
```