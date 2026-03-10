# Architecture Deep-Dive

This document explains the design decisions, physics intuitions, and component interactions in the Lightweight Gravitational Transformer (LGT).

---

## Table of Contents

1. [Motivation and Design Philosophy](#1-motivation-and-design-philosophy)
2. [Gravitational Attention Mechanism](#2-gravitational-attention-mechanism)
3. [Positional Encoding Strategies](#3-positional-encoding-strategies)
4. [Transformer Block Structure](#4-transformer-block-structure)
5. [Containment and Safety System](#5-containment-and-safety-system)
6. [VictorOS Cognitive Runtime](#6-victoros-cognitive-runtime)
7. [Tri-Model Architecture](#7-tri-model-architecture)
8. [Edge Deployment Pipeline](#8-edge-deployment-pipeline)
9. [Parameter Count and Memory](#9-parameter-count-and-memory)
10. [Design Trade-offs](#10-design-trade-offs)

---

## 1. Motivation and Design Philosophy

### Why Replace Standard Attention?

Standard scaled dot-product attention computes:

```
A_ij = softmax(q_i · k_j / √d)
```

This has several limitations in resource-constrained settings:
1. It requires three projections (Q, K, V), tripling the computation relative to the value projection alone.
2. The uniform softmax normalisation treats all tokens equally by default; distance and relevance must be learned from scratch.
3. There is no physical interpretability — it is hard to reason about *why* two tokens attend to each other.

### The Gravitational Alternative

LGT replaces the above with Newton's law of gravitation applied to learned token masses and curved positional coordinates:

```
F_ij = G · m_i · m_j / dist²(p_i, p_j)
A = softmax(F)
```

This provides:
- **Inductive bias**: Tokens that are close in positional space and have large masses naturally attract each other strongly — this is physically intuitive.
- **Fewer projections**: Only a single scalar `mass_proj` is needed per head (1 linear layer vs 3).
- **Interpretability**: You can directly inspect masses and forces to understand what the model is "doing".
- **Stable by construction**: The `event_horizon` and `max_force` (Hawking regularisation) provide hard bounds on attention values.

### Lightweight Design

The FFN uses a 2× expansion factor rather than the standard 4×. Combined with the reduced attention projections, this halves the parameter count relative to a standard transformer of the same depth and width.

---

## 2. Gravitational Attention Mechanism

### Force Computation

```
┌─────────────────────────────────────────────────────────────────┐
│  Input x ∈ ℝ^{batch × seq × head_dim}                          │
│                                                                  │
│  1. Token masses:  m = softplus(Wₘ x)   ∈ ℝ^{batch × seq × 1} │
│     (strictly positive; Wₘ ∈ ℝ^{1 × head_dim})                 │
│                                                                  │
│  2. Distance:  Δp_{ij} = p_i − p_j                             │
│     dist²_{ij} = ‖Δp‖² + ε                                     │
│     (+ curvature correction if curvature ≠ 0)                  │
│                                                                  │
│  3. Force:  F_{ij} = |G| · m_i · m_j / dist²_{ij}             │
│                                                                  │
│  4. Hawking cap:  F_{ij} = clamp(F_{ij}, max=max_force)        │
│                                                                  │
│  5. Weights:  α = softmax(F, dim=-1)                            │
│                                                                  │
│  6. Output:  out = α · x                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Curvature Modulation

When `curvature ≠ 0`, the effective distance is modulated:

```python
dist_norm = sqrt(dist_sq + event_horizon)
dist_sq   = dist_sq * (1 + curvature * cos(dist_norm))
```

This introduces a periodic ripple in the distance metric, creating a curved spacetime where tokens at certain distances are "closer" than Euclidean geometry would suggest. Higher curvature amplifies this effect.

### Multi-Head G Decay

Each successive head is initialised with a slightly lower `G`:

```python
G_h = G_base * (0.9 ** head_index)
```

This means head 0 uses strong gravitational coupling (coarse, long-range attention) while later heads use weaker coupling (fine-grained, local attention) — analogous to multi-scale feature extraction.

---

## 3. Positional Encoding Strategies

### CurvedPositionEmbedding

The default strategy. Positions are randomly initialised and learned end-to-end, applying a curvature modulation at inference:

```python
positions = Wₚ[:seq_len]                       # learnable, shape [seq, dim_pos]
curved    = positions * (1 + κ * sin(0.1 * positions))
```

The learnable curvature scale `κ` controls how aggressively the manifold bends. This gives the model full freedom to learn any geometry from data.

### FractalPositionEmbedding

An alternative strategy using a pre-computed power-law frequency basis:

```
ω_k = base_freq × scale_factor^(k × fractal_dim)
```

Band `k` contributes `dim_position / num_scales` sin/cos dimensions. The resulting embedding has self-similar structure at multiple scales, providing an inductive bias for:
- Hierarchical patterns (e.g., syntax in language)
- Long-range dependencies (the fractal spectrum covers many scales simultaneously)
- Periodic and quasi-periodic signals

A small learned residual allows the model to deviate from the pure fractal basis.

---

## 4. Transformer Block Structure

```
Input x [batch, seq, dim]
    │
    ├── MultiHeadGravitationalAttention ──┐
    │        (4 heads, each with own G)   │
    │                                     │ residual
    └─────────────────────────────────────┘
    │
    LayerNorm
    │
    ├── Lightweight FFN ─────────────────┐
    │   Linear(dim → 2×dim) + GELU       │
    │   Dropout                          │ residual
    │   Linear(2×dim → dim)              │
    │   Dropout                          │
    └───────────────────────────────────-┘
    │
    LayerNorm
    │
Output x [batch, seq, dim]
```

### Layer Depth and G Decay

Across the full stack of `num_layers` blocks, `G` decays as:

```
G_layer_i = G_base * (0.9 ** i)
```

Combined with the per-head decay above, the deepest layers use very small G values, effectively reverting to a softmax-uniform attention pattern — the model uses strong gravitational coupling only where useful (shallow layers for structure extraction) and weak coupling in deep layers (refinement).

---

## 5. Containment and Safety System

The safety system operates at three levels:

### Level 1: Hawking Regularisation (per attention head)

```python
forces = forces.clamp(max=max_force)   # default 50.0
```

Prevents any single token pair from dominating attention (gravitational collapse prevention).

### Level 2: ContainmentProtocol (per training step)

After each backward pass:

1. **Gradient clipping**: `clip_grad_norm_(params, max_grad_norm)` — prevents gradient explosions.
2. **Attention-force dampening**: If mean force > `max_attention_force`, reduce all `G` parameters by 10%.
3. **Bekenstein entropy penalty**: Adds `λ × H` to the loss, where `H` is the Gaussian entropy upper bound of the layer outputs — prevents information spreading.
4. **Divergence/collapse detection**: Halt training if `loss > max_loss` (diverged) or `loss < min_loss` (collapsed).
5. **Architecture proposals**: When `stability_ema > 0.95`, propose adding a layer or increasing curvature.

### Level 3: Mirror Layer (per forward pass)

The `MirrorLayer` monitors the rolling mean gravitational force and maintains a stability score:

```
stability = 1 / (1 + mean_force / max_force_threshold)
```

When force exceeds the threshold, it calls `correction_callback` and logs to the Ledger. This is designed for the VictorOS Cortex to apply corrections at runtime without modifying training code.

---

## 6. VictorOS Cognitive Runtime

```
VictorOS Cortex
    │
    ├── @victoros_module annotation
    │       │
    │       └── Attaches VictorOSModuleMetadata to class
    │           Wraps __init__ to auto-provision Ledger + MirrorLayer
    │
    ├── Ledger
    │       │
    │       ├── Append-only in-memory buffer
    │       ├── JSONL persistence (tamper-evident audit trail)
    │       └── Events: inference, checkpoint, containment_stop,
    │                   grad_clip, attention_dampening, mirror_layer,
    │                   containment_correction, architecture_proposal,
    │                   train_step, eval_step, meta_curvature_update
    │
    ├── MirrorLayer
    │       │
    │       ├── Receives per-layer diagnostics via callback
    │       ├── Computes rolling stability score
    │       └── Emits correction signals when threshold exceeded
    │
    └── Architecture Proposals
            │
            ├── Generated by ContainmentProtocol or LGTVictorOSModule
            ├── Format: {"change": "add_layer"|"increase_curvature", ...}
            └── Must be applied externally (by Cortex or training script)
```

### Event Types

| Event | Source | Payload Keys |
|---|---|---|
| `inference` | `LGTVictorOSModule.process()` | `seq_len`, `stability_score`, `corrections`, `output_mean`, `output_std` |
| `snapshot` | `LGTVictorOSModule.get_snapshot()` | `model_config` |
| `architecture_proposal` | `ContainmentProtocol`, `LGTVictorOSModule` | `change`, `new_num_layers` or `new_curvature`, `reason` |
| `mirror_layer` | `MirrorLayer.__call__()` | `layer`, `mean_force`, `mean_mass`, `stability_score` |
| `containment_correction` | `MirrorLayer.__call__()` | `layer`, `trigger`, `value`, `threshold`, `correction` |
| `containment_stop` | `ContainmentProtocol.step()` | `reason`, `loss` |
| `grad_clip` | `ContainmentProtocol.step()` | `total_norm` |
| `attention_dampening` | `ContainmentProtocol.step()` | `mean_force`, `threshold` |
| `train_step` | `TrainingLoop.train_step()` | `step`, `loss`, `stability` |
| `eval_step` | `TrainingLoop.fit()` | `step`, `val_loss` |
| `meta_curvature_update` | `TrainingLoop.eval_step()` | `updates` |

---

## 7. Tri-Model Architecture

```
WorldInput  [batch, seq_w, dim] ──► WorldModel  (curvature=0.25, G=1.0) ──► world_out
SelfInput   [batch, seq_s, dim] ──► SelfModel   (curvature=0.15, G=0.8) ──► self_out
EnvInput    [batch, seq_e, dim] ──► EnvModel    (curvature=0.10, G=1.2) ──► env_out
                                         │
                              Sequence alignment (zero-pad to max_len)
                                         │
                          ┌──────────────▼──────────────┐
                          │  CrossGravitationalFusion    │
                          │                              │
                          │  w_mass = softplus(Ww·world̄) │
                          │  s_mass = softplus(Ws·self̄)  │
                          │  e_mass = softplus(We·ēnv)   │
                          │                              │
                          │  World cross-attends to      │
                          │    G·s_mass·self + G·e_mass·env
                          │  Self cross-attends to       │
                          │    G·w_mass·world + G·e_mass·env
                          │  Env cross-attends to        │
                          │    G·w_mass·world + G·s_mass·self
                          └──────────────┬──────────────┘
                                         │
                           cat([world_fused, self_fused, env_fused])
                                         │
                                   LayerNorm
                                         │
                             Linear(3·dim → output_dim)
                                         │
                                      Output
```

### Sub-Model Tuning Rationale

| Sub-model | Curvature | G | Intuition |
|---|---|---|---|
| WorldModel | 0.25 | 1.0 | External context needs high curvature to capture long-range semantic structure |
| SelfModel | 0.15 | 0.8 | Internal state is more uniform; moderate coupling |
| EnvironmentModel | 0.10 | 1.2 | Urgency/salience requires strong gravitational pull but flat positional geometry |

---

## 8. Edge Deployment Pipeline

```
build_model(preset)
    │
    ├── Optional: quantize_dynamic(model, "int8" | "float16")
    │       │
    │       ├── "int8": torch.ao.quantization.quantize_dynamic({nn.Linear})
    │       │   → ~4× size reduction, faster CPU inference
    │       └── "float16": model.half()
    │           → ~2× size reduction, GPU/NPU speedup
    │
    ├── save_checkpoint(model, path, metadata)
    │   → .pt file with state_dict + config metadata
    │
    └── export_torchscript(model, example, path)
        │
        ├── FP32/FP16: torch.jit.trace(model, example_input)
        │   → portable, inference-optimised TorchScript
        └── INT8: torch.jit.script(model)
            → script instead of trace for quantised models
```

### Memory Footprints

| Preset | FP32 | FP16 | INT8 |
|---|---|---|---|
| edge_150k | ~0.6 MB | ~0.3 MB | ~0.15 MB |
| meta_probe | ~2.3 MB | ~1.1 MB | ~0.6 MB |
| victorcos | ~5.3 MB | ~2.7 MB | ~1.3 MB |
| fractal_res | ~8.0 MB | ~4.0 MB | ~2.0 MB |

---

## 9. Parameter Count and Memory

### Breakdown per Block (dim=128, heads=4, ff_expansion=2)

| Component | Parameters |
|---|---|
| `mass_proj` per head | `head_dim = 32` |
| `G` per head | `1` |
| `out_proj` | `128 × 128 = 16,384` |
| FFN `Linear(128→256)` + `Linear(256→128)` | `128×256 + 256×128 = 65,536` |
| LayerNorm ×2 | `2 × 2 × 128 = 512` |
| `token_mass` (per-token context) | `128` |
| **Block total** | **~82,700** |

For `num_layers=4`: ~330 K per block stack + position embeddings (~32 K) ≈ **600 K** total (meta_probe preset).

---

## 10. Design Trade-offs

| Decision | Trade-off |
|---|---|
| Gravitational vs QKV attention | Lower parameter count; loses the expressive power of independent Q, K, V projections |
| `mass_proj` (scalar mass) vs full Q/K projections | Very lightweight; can only represent token importance as a scalar, not a vector |
| `curvature` modulation | Adds non-linearity to distances but may be harder to optimise than linear distances |
| 2× FFN expansion (vs standard 4×) | Halves FFN parameters; may reduce capacity on complex tasks |
| Per-layer G decay | Provides multi-scale bias; removes the possibility of uniform G across layers |
| `max_force` Hawking cap | Prevents collapse but could prevent the model from learning very sharp attention patterns |
| `tie_weights` (LM head = embedding) | Reduces parameters by ~`vocab_size × dim`; standard in language models |
