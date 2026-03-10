# User Guide

This guide walks through the most common use-cases for the Lightweight Gravitational Transformer (LGT), from basic inference to full training with containment, VictorOS integration, edge export, and the tri-model architecture.

---

## Table of Contents

1. [Concepts](#1-concepts)
2. [Basic Inference](#2-basic-inference)
3. [Language Modelling](#3-language-modelling)
4. [Fractal Position Embeddings](#4-fractal-position-embeddings)
5. [Attention Diagnostics](#5-attention-diagnostics)
6. [Training with ContainmentProtocol](#6-training-with-containmentprotocol)
7. [MetaCurvatureScheduler](#7-metacurvaturescheduler)
8. [Ledger and Audit Trail](#8-ledger-and-audit-trail)
9. [Mirror Layer](#9-mirror-layer)
10. [VictorOS Integration](#10-victoros-integration)
11. [Tri-Model Architecture](#11-tri-model-architecture)
12. [Edge Export and Deployment](#12-edge-export-and-deployment)
13. [Tips and Best Practices](#13-tips-and-best-practices)

---

## 1. Concepts

### Gravitational Attention

Standard transformers compute attention as:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

LGT replaces this with a force-based computation:

```
F_ij = G · m_i · m_j / (dist(p_i, p_j)² + ε)
Attention = softmax(F) · X
```

where:
- **`m_i`** is a learnable scalar mass for token `i` (always positive via `softplus`)
- **`p_i`** is a positional vector in a curved or fractal manifold
- **`G`** is a learnable gravitational constant (one per attention head)
- **`ε`** (`event_horizon`) prevents division by zero
- The **`max_force`** (Hawking regularisation) caps the maximum force to prevent attention collapse

### Curvature

The `curvature` parameter applies a non-linear modulation to inter-token distances:

```
dist_sq *= (1 + curvature * cos(||p||))
```

This creates a curved spacetime in which close tokens in positional space exert disproportionately large gravitational pull, and far tokens are further attenuated.

### Bekenstein Entropy Penalty

To prevent the model from encoding too much information in a single representation (information spreading), the training loop can add an entropy regularisation term:

```
H ≈ 0.5 · log(2π·e·var(x))   (Gaussian entropy upper bound)
loss += λ · H
```

This encourages compressed, information-efficient representations analogous to the Bekenstein-Hawking entropy bound.

---

## 2. Basic Inference

```python
import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

# Continuous-embedding model (no vocabulary)
model = LightweightGravitationalTransformer(
    dim_model=128,       # embedding dimension
    dim_position=64,     # position vector dimension
    num_layers=4,        # number of gravitational blocks
    num_heads=4,         # attention heads per block
    max_seq_len=512,
    curvature=0.15,
    dropout=0.0,         # set to 0 for inference
)
model.eval()

# Batch of continuous embeddings: [batch, seq_len, dim_model]
x = torch.randn(2, 32, 128)

with torch.no_grad():
    output, diagnostics = model(x, return_diagnostics=True)

print(output.shape)               # [2, 32, 128]
print(diagnostics["curvature"])   # 0.15
```

### Precomputed Positions

You can supply your own position vectors (e.g., from an external geometry):

```python
custom_positions = torch.randn(32, 64)   # [seq_len, dim_position]
output, _ = model(x, positions=custom_positions)
```

---

## 3. Language Modelling

```python
import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

model = LightweightGravitationalTransformer(
    vocab_size=32000,
    dim_model=256,
    num_layers=6,
    num_heads=8,
    max_seq_len=512,
    tie_weights=True,    # share embedding and output-projection weights
)

# Token IDs: [batch, seq_len]
token_ids = torch.randint(0, 32000, (4, 64))
logits, _ = model(token_ids)
print(logits.shape)   # [4, 64, 32000]

# Greedy decoding
predicted_ids = logits.argmax(dim=-1)
print(predicted_ids.shape)   # [4, 64]
```

### Autoregressive Generation

```python
def generate(model, prompt_ids, max_new_tokens=50, temperature=1.0):
    model.eval()
    ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(ids[:, -model.pos_embedding.positions.shape[0]:])
            next_logits = logits[:, -1, :] / temperature
            next_id = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
            ids = torch.cat([ids, next_id], dim=-1)
    return ids

prompt = torch.tensor([[1, 42, 17, 500]])   # [batch=1, seq=4]
generated = generate(model, prompt, max_new_tokens=20)
print(generated.shape)   # [1, 24]
```

---

## 4. Fractal Position Embeddings

Use `use_fractal_positions=True` to replace the default curved positions with a fractal power-law spectrum:

```python
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

model = LightweightGravitationalTransformer(
    dim_model=128,
    use_fractal_positions=True,
    fractal_dim=1.5,   # Hausdorff dimension: >1 compresses high-freq scales
)

x = torch.randn(1, 64, 128)
output, _ = model(x)
```

Use `FractalPositionEmbedding` directly:

```python
from fractal_position_embedding import FractalPositionEmbedding

embed = FractalPositionEmbedding(
    max_seq_len=512,
    dim_position=64,
    fractal_dim=1.5,
    num_scales=4,          # number of frequency bands
    learnable_residual=True,
)

positions = embed(seq_len=32)   # [32, 64]
```

### Choosing Between Curved and Fractal Positions

| Property | `CurvedPositionEmbedding` | `FractalPositionEmbedding` |
|---|---|---|
| Basis | Learnable random init | Sinusoidal power-law |
| Multi-scale | No | Yes (`num_scales`) |
| Inductive bias | General manifold | Self-similar structure |
| Parameters | `max_seq_len × dim_position` | `2 + max_seq_len × dim_position` residual |
| Best for | General tasks | Long-range, hierarchical patterns |

---

## 5. Attention Diagnostics

### Per-Layer Diagnostics

```python
output, diagnostics = model(x, return_diagnostics=True)

for layer_info in diagnostics["layers"]:
    print(f"Layer {layer_info['layer']}:")
    print(f"  mean_force = {layer_info['mean_force']:.4f}")
    print(f"  mean_mass  = {layer_info['mean_mass']:.4f}")
    print(f"  hawking_limit = {layer_info['hawking_limit']}")
```

### Per-Head Diagnostics

```python
from gravitational_attention import MultiHeadGravitationalAttention

attn = MultiHeadGravitationalAttention(dim_model=128, num_heads=4)
x = torch.randn(2, 16, 128)
diag = attn.get_attention_diagnostics(x)

for head, stats in diag.items():
    print(f"{head}: G={stats['G']:.4f}, mean_mass={stats['mean_mass']:.4f}, "
          f"mean_force={stats['mean_force']:.4f}")
```

### Attention Snapshot (for Ledger tracing)

```python
snapshot = model.get_attention_snapshot(x)
# snapshot contains model_config + per-layer attention metrics
```

---

## 6. Training with ContainmentProtocol

The `ContainmentProtocol` acts as a safety wrapper around the standard training loop:

```python
import torch
import torch.nn as nn
from lightweight_gravitational_transformer import LightweightGravitationalTransformer
from training import ContainmentConfig, ContainmentProtocol

model = LightweightGravitationalTransformer(vocab_size=1000, dim_model=128)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
config = ContainmentConfig(
    max_grad_norm=1.0,
    max_attention_force=40.0,
    bekenstein_lambda=1e-4,
)
protocol = ContainmentProtocol(config=config, model=model)

# Training step
model.train()
x = torch.randint(0, 1000, (4, 32))
y = torch.randint(0, 1000, (4, 32))
logits, diagnostics = model(x, return_diagnostics=True)
loss = loss_fn(logits.view(-1, 1000), y.view(-1))

# Optional: add Bekenstein entropy penalty
loss = loss + protocol.bekenstein_penalty(logits)
loss.backward()

# ContainmentProtocol checks happen AFTER backward, BEFORE optimizer.step()
summary = protocol.step(loss, diagnostics)

if summary["stopped"]:
    print("Training halted:", summary)
else:
    optimizer.step()
    optimizer.zero_grad()

print(f"Stability EMA: {summary['stability']:.3f}")
print(f"Proposal: {summary['proposal']}")
```

### Using `TrainingLoop` (All-in-One)

```python
from training import TrainingLoop, TrainingConfig, ContainmentConfig
from victorcos_module import Ledger

ledger = Ledger(agent_id="run_001", persist_path="logs/run_001.jsonl")

loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    loss_fn=lambda logits, y: loss_fn(logits.view(-1, logits.size(-1)), y.view(-1)),
    config=TrainingConfig(
        max_steps=10_000,
        eval_every=500,
        use_bekenstein_penalty=True,
        use_meta_curvature=True,
        grad_accumulation_steps=4,   # gradient accumulation
    ),
    containment_config=ContainmentConfig(),
    ledger=ledger,
)

def data_gen(vocab=1000, seq=32, batch=8):
    while True:
        yield torch.randint(0, vocab, (batch, seq)), torch.randint(0, vocab, (batch, seq))

summary = loop.fit(
    data_gen(),
    on_proposal=lambda p: print("Architecture proposal:", p),
)
ledger.flush()
print(summary)
```

---

## 7. MetaCurvatureScheduler

Adjusts per-layer curvature parameters based on validation loss direction:

```python
from training import MetaCurvatureScheduler

scheduler = MetaCurvatureScheduler(
    model=model,
    lr=0.01,           # meta-learning rate
    min_curvature=0.0,
    max_curvature=0.5,
)

# Call after each validation evaluation
val_loss = 2.34
updates = scheduler.step(val_loss)
print(updates)   # {"pos_embedding.curvature_scale": 0.152, ...}
```

---

## 8. Ledger and Audit Trail

The `Ledger` provides a tamper-evident, human-readable event log:

```python
from victorcos_module import Ledger

ledger = Ledger(
    agent_id="my_agent",
    persist_path="logs/agent.jsonl",  # JSONL format; None for memory-only
    max_memory_entries=1000,          # auto-flush threshold
)

# Log any structured event
ledger.log("inference", {"seq_len": 32, "stability": 0.98})
ledger.log("checkpoint", {"path": "ckpt_step1000.pt"})

# Query in-memory entries
all_entries = ledger.entries()
inference_entries = ledger.entries(event_filter="inference")
print(f"Total entries: {len(ledger)}")

# Get a serialisable snapshot
snapshot = ledger.snapshot()

# Flush to disk (appends to JSONL file)
n_flushed = ledger.flush()
print(f"Flushed {n_flushed} entries")
```

### Reading JSONL Logs

```python
import json

with open("logs/agent.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        print(entry["event"], entry["timestamp"], entry["payload"])
```

---

## 9. Mirror Layer

The `MirrorLayer` sits between the model's forward pass and the VictorOS Cortex, monitoring stability in real time:

```python
from victorcos_module import Ledger, MirrorLayer
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

model = LightweightGravitationalTransformer(dim_model=128, num_layers=4)
ledger = Ledger(agent_id="mirror_test")

corrections = []

mirror = MirrorLayer(
    ledger=ledger,
    max_force_threshold=40.0,
    stability_window=20,
    correction_callback=lambda layer_idx, correction_type: corrections.append({
        "layer": layer_idx, "correction": correction_type
    }),
)

# Pass mirror as the callback in forward()
x = torch.randn(1, 16, 128)
output, _ = model(x, return_diagnostics=True, mirror_layer_callback=mirror)

print(f"Stability score: {mirror.stability_score():.3f}")
print(f"Corrections triggered: {corrections}")
```

When `mean_force > max_force_threshold`, the Mirror Layer:
1. Logs a `containment_correction` event to the Ledger.
2. Calls `correction_callback` with `(layer_idx, "attention_dampening")`.

---

## 10. VictorOS Integration

### Using `LGTVictorOSModule`

```python
import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer
from victorcos_module import LGTVictorOSModule

model = LightweightGravitationalTransformer(dim_model=128, num_layers=4)

module = LGTVictorOSModule(
    model=model,
    agent_id="lgt_core_v1",
    persist_path="ledger/core.jsonl",
    max_force_threshold=40.0,
)

x = torch.randn(1, 32, 128)
result = module.process(x)
print(result["stability"])        # float in [0, 1]
print(result["output"].shape)     # [1, 32, 128]

# Architecture self-evolution proposal
proposal = module.propose_architecture_change(
    current_config={"num_layers": 4, "curvature": 0.15},
    stability_threshold=0.95,
)
if proposal:
    print("Proposal:", proposal)
    # {"change": "increase_curvature", "new_curvature": 0.165, "reason": "..."}
```

### Custom Module with `@victoros_module`

```python
from victorcos_module import victoros_module, VictorOSBaseModule

@victoros_module(
    name="specialized_lgt",
    version="1.0.0",
    requirements=["torch>=2.0.0"],
    containment_native=True,
    description="Domain-specialised LGT module.",
)
class SpecialisedLGT(VictorOSBaseModule):
    def __init__(self, dim_model=256):
        from lightweight_gravitational_transformer import LightweightGravitationalTransformer
        self.model = LightweightGravitationalTransformer(
            dim_model=dim_model,
            use_fractal_positions=True,
        )

    def process(self, x):
        output, _ = self.model(
            x,
            return_diagnostics=True,
            mirror_layer_callback=self.mirror_layer,
        )
        self.ledger.log("inference", {
            "shape": list(x.shape),
            "stability": self.mirror_layer.stability_score(),
        })
        return output

agent = SpecialisedLGT(dim_model=256)
print(agent._victoros_meta.name)    # "specialized_lgt"
print(len(agent.ledger))            # 0 (empty on init)
```

### Checkpointing

```python
# Save
module.save_checkpoint("checkpoints/module_step1000.pt", extra={"step": 1000})

# Load
state = module.load_checkpoint("checkpoints/module_step1000.pt")
print(state["extra"])    # {"step": 1000}
```

---

## 11. Tri-Model Architecture

The `TriModelTransformer` processes three input streams in parallel and fuses them via cross-gravitational attention:

```python
import torch
from tri_model import TriModelTransformer

tri = TriModelTransformer(
    dim_model=128,
    dim_position=64,
    num_layers=4,
    num_heads=4,
    vocab_size=32000,    # set if inputs are token IDs
    max_seq_len=256,
    output_dim=128,
)

# Token IDs (or continuous embeddings if vocab_size=None)
world = torch.randint(0, 32000, (2, 32))   # external context
self_ = torch.randint(0, 32000, (2, 16))   # internal state
env   = torch.randint(0, 32000, (2, 8))    # interaction urgency

output, diagnostics = tri(world, self_, env, return_diagnostics=True)
print(output.shape)                        # [2, 32, 128]
print(diagnostics["fusion"]["world_G"])    # gravitational constant of fusion layer

# Full VictorOS causal trace snapshot
snapshot = tri.get_tri_snapshot(world, self_, env)
```

### Stream-Specific Parameters

| Stream | Curvature | G | Semantic Role |
|---|---|---|---|
| WorldModel | 0.25 (high) | 1.0 | External semantic context |
| SelfModel | 0.15 (medium) | 0.8 | Agent internal state |
| EnvironmentModel | 0.10 (low) | 1.2 | Interaction urgency |

---

## 12. Edge Export and Deployment

### Exporting a Model

```python
from export_edge_model import export_edge_model

# Export edge_150k preset with INT8 quantisation
paths = export_edge_model(
    config_name="edge_150k",
    vocab_size=32000,
    max_seq_len=512,
    quantize="int8",
    output_dir="exported_models",
)

print(paths["checkpoint"])    # exported_models/lgt_edge_150k_int8.pt
print(paths["torchscript"])   # exported_models/lgt_edge_150k_int8_traced.pt
print(paths["config"])        # {"n_params": ..., "vocab_size": ..., ...}
```

### CLI Export

```bash
# Smallest model, INT8 quantisation
python export_edge_model.py --config edge_150k --quantize int8 --output-dir models/

# VictorOS preset, FP16
python export_edge_model.py --config victorcos --quantize float16

# Full fractal model, no quantisation
python export_edge_model.py --config fractal_res --quantize none --fractal-positions
```

### Loading an Exported Checkpoint

```python
import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer
from export_edge_model import build_model, PRESETS

# Rebuild model from preset and load weights
meta = torch.load("exported_models/lgt_edge_150k.pt", weights_only=False)
model = build_model(
    config_name=meta["metadata"]["config"],
    vocab_size=meta["metadata"]["vocab_size"],
)
model.load_state_dict(meta["model_state_dict"])
model.eval()
```

### Loading a TorchScript Model

```python
import torch

scripted_model = torch.jit.load("exported_models/lgt_edge_150k_traced.pt")
x = torch.randint(0, 32000, (1, 32))
output = scripted_model(x)    # returns (logits, None)
```

---

## 13. Tips and Best Practices

### Choosing Model Size

| Use Case | Recommended Preset | `dim_model` | Params |
|---|---|---|---|
| Microcontroller / very low power | `edge_150k` | 64 | ~150 K |
| Raspberry Pi / mobile | `meta_probe` | 128 | ~600 K |
| VictorOS cognitive agent | `victorcos` | 192 | ~1.4 M |
| Research / full quality | `fractal_res` | 256 | ~2.1 M |

### Stability Tuning

- Start with `curvature=0.15` and adjust based on training stability.
- If attention forces diverge (> `max_force`), reduce `gravitational_constant` or lower `max_force`.
- Enable `use_bekenstein_penalty=True` in `TrainingConfig` to prevent representation collapse.
- Monitor `stability_score` from the `MirrorLayer`; values < 0.5 indicate runaway dynamics.

### Gravitational Constant Decay

By default, `G` decays across layers as `G × 0.9^layer_index`. This means:
- Early layers use strong gravitational attraction (coarse structure).
- Later layers use weaker forces (fine-grained refinement).

You can customise the decay by instantiating `LightweightGravitationalBlock` directly.

### Memory Efficiency

- Use `dropout=0.0` during inference for a small speedup.
- Use `return_diagnostics=False` unless you need introspection (avoids extra computation).
- For batch inference, increase batch size before sequence length.

### Debugging NaN / Inf

If you encounter NaN values:
1. Check that `event_horizon > 0` (prevents division by zero in force computation).
2. Lower `gravitational_constant` (default 1.0) to reduce initial force magnitudes.
3. Enable `max_force` (Hawking regularisation) to prevent force blow-up.
4. Reduce learning rate and enable gradient clipping via `ContainmentConfig(max_grad_norm=1.0)`.
