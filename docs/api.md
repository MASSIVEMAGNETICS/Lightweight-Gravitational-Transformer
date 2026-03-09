# API Reference

Complete reference for all public classes and functions in the Lightweight Gravitational Transformer library.

---

## Table of Contents

- [gravitational_attention](#gravitational_attention)
  - [GravitationalAttentionHead](#gravitationalattentionhead)
  - [MultiHeadGravitationalAttention](#multiheadgravitationalattention)
- [fractal_position_embedding](#fractal_position_embedding)
  - [FractalPositionEmbedding](#fractalpositionembedding)
- [lightweight_gravitational_transformer](#lightweight_gravitational_transformer)
  - [CurvedPositionEmbedding](#curvedpositionembedding)
  - [LightweightGravitationalBlock](#lightweightgravitationalblock)
  - [LightweightGravitationalTransformer](#lightweightgravitationaltransformer)
- [victorcos_module](#victorcos_module)
  - [LedgerEntry](#ledgerentry)
  - [Ledger](#ledger)
  - [MirrorLayer](#mirrorlayer)
  - [VictorOSModuleMetadata](#victorosmodulemetadata)
  - [victoros_module (decorator)](#victoros_module-decorator)
  - [VictorOSBaseModule](#victorosbasemodule)
  - [LGTVictorOSModule](#lgtvictorosmodule)
- [training](#training)
  - [ContainmentConfig](#containmentconfig)
  - [ContainmentProtocol](#containmentprotocol)
  - [MetaCurvatureScheduler](#metacurvaturescheduler)
  - [TrainingConfig](#trainingconfig)
  - [TrainingLoop](#trainingloop)
- [tri_model](#tri_model)
  - [CrossGravitationalFusion](#crossgravitationalfusion)
  - [TriModelTransformer](#trimodeltransformer)
- [export_edge_model](#export_edge_model)
  - [PRESETS](#presets)
  - [build_model](#build_model)
  - [export_torchscript](#export_torchscript)
  - [quantize_dynamic](#quantize_dynamic)
  - [save_checkpoint](#save_checkpoint)
  - [export_edge_model (function)](#export_edge_model-function)

---

## `gravitational_attention`

### `GravitationalAttentionHead`

```python
class GravitationalAttentionHead(nn.Module)
```

Single head of gravitational attention. Computes attention weights from gravitational forces between tokens.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `head_dim` | `int` | — | Dimension of each head slice |
| `gravitational_constant` | `float` | `1.0` | Initial value of the learnable `G` |
| `event_horizon` | `float` | `1e-6` | Minimum effective distance² (prevents ÷0) |
| `max_force` | `float \| None` | `50.0` | Hawking regularisation cap; `None` disables |
| `curvature` | `float` | `0.15` | Curvature applied to inter-token distances |

**Learnable Parameters**

| Name | Shape | Description |
|---|---|---|
| `G` | scalar | Per-head gravitational constant |
| `mass_proj.weight` | `[1, head_dim]` | Linear projection: head slice → scalar mass |

**Methods**

#### `forward(x, positions=None) → (Tensor, Tensor)`

| Argument | Shape | Description |
|---|---|---|
| `x` | `[batch, seq, head_dim]` | Token representations |
| `positions` | `[seq, dim_pos]` or `None` | Curved/fractal position vectors |

Returns `(output, masses)` where `output` is `[batch, seq, head_dim]` and `masses` is `[batch, seq]`.

---

### `MultiHeadGravitationalAttention`

```python
class MultiHeadGravitationalAttention(nn.Module)
```

Multi-head gravitational attention with optional independent `G` per head.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim_model` | `int` | — | Total model dimension; must be divisible by `num_heads` |
| `dim_position` | `int` | `64` | Positional vector dimension (informational) |
| `num_heads` | `int` | `4` | Number of attention heads |
| `gravitational_constant` | `float` | `1.0` | Initial G (decayed per head as `G × 0.9^h` when `different_G_per_head=True`) |
| `event_horizon` | `float` | `1e-6` | Minimum distance² |
| `max_force` | `float \| None` | `50.0` | Hawking cap |
| `curvature` | `float` | `0.15` | Spacetime curvature |
| `different_G_per_head` | `bool` | `True` | Give each head an independent learnable `G` |

**Methods**

#### `forward(x, positions=None) → Tensor`

Returns `[batch, seq, dim_model]`.

#### `get_attention_diagnostics(x, positions=None) → Dict[str, Dict[str, float]]`

Returns per-head statistics. Accepts NumPy arrays or PyTorch tensors.

```python
{
    "head_0": {"mean_mass": float, "mean_force": float, "G": float, "curvature": float},
    "head_1": {...},
    ...
}
```

---

## `fractal_position_embedding`

### `FractalPositionEmbedding`

```python
class FractalPositionEmbedding(nn.Module)
```

Multi-scale sinusoidal position embedding with power-law (fractal) frequency spacing.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_seq_len` | `int` | — | Maximum sequence length |
| `dim_position` | `int` | — | Output position vector dimension |
| `num_scales` | `int` | `4` | Number of frequency bands |
| `fractal_dim` | `float` | `1.5` | Hausdorff-like dimension: `ω_k = base_freq × scale_factor^(k × fractal_dim)` |
| `base_freq` | `float` | `1.0` | Lowest (coarsest) frequency |
| `scale_factor` | `float` | `2.0` | Multiplicative step between adjacent bands |
| `learnable_residual` | `bool` | `True` | Add a learned residual offset per position |

**Buffers / Parameters**

| Name | Shape | Description |
|---|---|---|
| `basis` | `[max_seq_len, dim_position]` | Pre-computed fractal sinusoidal basis (buffer) |
| `scale` | scalar | Learnable overall scale for the basis |
| `residual` | `[max_seq_len, dim_position]` | Learned per-position residual (if `learnable_residual=True`) |
| `curvature` | scalar | Learnable curvature modulation |

**Methods**

#### `forward(seq_len) → Tensor`

Returns `[seq_len, dim_position]`.

---

## `lightweight_gravitational_transformer`

### `CurvedPositionEmbedding`

```python
class CurvedPositionEmbedding(nn.Module)
```

Learnable positions on a curved manifold. Default position encoding when `use_fractal_positions=False`.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_seq_len` | `int` | — | Maximum sequence length |
| `dim_position` | `int` | — | Position vector dimension |
| `curvature` | `float` | `0.15` | Initial curvature scale (learnable) |

#### `forward(seq_len) → Tensor`

Returns `[seq_len, dim_position]`.

---

### `LightweightGravitationalBlock`

```python
class LightweightGravitationalBlock(nn.Module)
```

Single transformer block: gravitational attention + lightweight FFN + layer norms.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim_model` | `int` | `128` | Model dimension |
| `dim_position` | `int` | `64` | Position vector dimension |
| `num_heads` | `int` | `4` | Attention heads |
| `ff_expansion` | `float` | `2.0` | FFN hidden dimension = `dim_model × ff_expansion` |
| `gravitational_constant` | `float` | `1.0` | Base G for this block |
| `curvature` | `float` | `0.15` | Spacetime curvature |
| `event_horizon` | `float` | `1e-6` | Minimum distance² |
| `max_force` | `float \| None` | `50.0` | Hawking cap |
| `dropout` | `float` | `0.1` | Dropout probability |
| `learnable_masses` | `bool` | `True` | Store per-token mass context as a parameter vs buffer |

**Methods**

#### `forward(x, positions=None, return_diagnostics=False) → (Tensor, Dict | None)`

| Argument | Description |
|---|---|
| `x` | `[batch, seq, dim_model]` |
| `positions` | `[seq, dim_position]` or `None` |
| `return_diagnostics` | If `True`, return a diagnostics dict |

Returns `(output, diagnostics)`. `diagnostics` contains:

```python
{
    "mean_force": float,
    "mean_mass": float,
    "curvature_active": bool,
    "hawking_limit": float | None,
    "seq_len": int,
    "per_head": {head_0: {...}, ...},
}
```

---

### `LightweightGravitationalTransformer`

```python
class LightweightGravitationalTransformer(nn.Module)
```

Complete transformer stack.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int \| None` | `None` | Vocabulary size; `None` for continuous input |
| `dim_model` | `int` | `128` | Model dimension |
| `dim_position` | `int` | `64` | Position vector dimension |
| `num_layers` | `int` | `4` | Number of gravitational blocks |
| `num_heads` | `int` | `4` | Heads per block |
| `max_seq_len` | `int` | `512` | Maximum sequence length |
| `curvature` | `float` | `0.15` | Spacetime curvature |
| `gravitational_constant` | `float` | `1.0` | Base G (decays as `G × 0.9^i` per layer) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `tie_weights` | `bool` | `False` | Tie embedding ↔ output projection |
| `use_fractal_positions` | `bool` | `False` | Use `FractalPositionEmbedding` |
| `fractal_dim` | `float` | `1.5` | Hausdorff dimension (fractal only) |

**Methods**

#### `forward(x, positions=None, return_diagnostics=False, mirror_layer_callback=None) → (Tensor, Dict | None)`

| Argument | Type | Description |
|---|---|---|
| `x` | Tensor | `[batch, seq, dim_model]` or token IDs `[batch, seq]` |
| `positions` | Tensor or `None` | Override position vectors `[seq, dim_pos]` |
| `return_diagnostics` | bool | Enable per-layer diagnostic output |
| `mirror_layer_callback` | callable or `None` | `callback(layer_idx, diag_dict)` |

Returns `(output, diagnostics)`.

**Diagnostics structure:**

```python
{
    "layers": [
        {"layer": 0, "mean_force": ..., "mean_mass": ..., ...},
        ...
    ],
    "curvature": float,
    "final_norm_stats": {"mean": float, "std": float},
}
```

#### `get_attention_snapshot(x) → Dict`

Generate a full attention snapshot for Ledger logging.

```python
{
    "timestamp": float | None,
    "model_config": {"dim_model": int, "curvature": float, "num_layers": int},
    "attention_metrics": diagnostics,
}
```

---

## `victorcos_module`

### `LedgerEntry`

```python
@dataclass
class LedgerEntry:
    entry_id: str      # UUID4
    timestamp: float   # UNIX timestamp
    agent_id: str
    event: str
    payload: Dict[str, Any]
```

**Methods:** `to_dict() → Dict`, `to_json() → str`

---

### `Ledger`

```python
class Ledger
```

Append-only structured event log.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent_id` | `str` | `"default"` | Owning agent identifier |
| `persist_path` | `str \| None` | `None` | Path to JSONL file; `None` = memory-only |
| `max_memory_entries` | `int` | `1000` | Auto-flush threshold |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `log(event, payload=None)` | `LedgerEntry` | Create and store a new entry |
| `flush()` | `int` | Write entries to disk; returns count flushed |
| `entries(event_filter=None)` | `List[LedgerEntry]` | Return in-memory entries, optionally filtered |
| `snapshot()` | `Dict` | All entries as a serialisable dict |
| `__len__()` | `int` | Number of in-memory entries |

---

### `MirrorLayer`

```python
class MirrorLayer
```

Real-time stability monitor.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ledger` | `Ledger \| None` | `None` | Auto-creates one if `None` |
| `max_force_threshold` | `float` | `40.0` | Force value triggering dampening correction |
| `stability_window` | `int` | `20` | Rolling window for stability score |
| `correction_callback` | `callable \| None` | `None` | `callback(layer_idx, correction_type)` |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `__call__(layer_idx, diag)` | `None` | Callback compatible with `mirror_layer_callback` |
| `stability_score()` | `float` | Rolling stability score in `[0, 1]` |

---

### `VictorOSModuleMetadata`

```python
@dataclass
class VictorOSModuleMetadata:
    name: str
    version: str
    requirements: List[str]
    containment_native: bool
    description: str
```

---

### `victoros_module` (decorator)

```python
def victoros_module(
    name: str,
    version: str,
    requirements: Optional[List[str]] = None,
    containment_native: bool = False,
    description: str = "",
) -> Callable[[Type], Type]
```

Class decorator. Attaches `_victoros_meta` metadata and wraps `__init__` to auto-provision a `Ledger` and `MirrorLayer`.

---

### `VictorOSBaseModule`

```python
class VictorOSBaseModule
```

Base class for VictorOS modules. Provides `ledger`, `mirror_layer`, `now()`, `save_checkpoint()`, `load_checkpoint()`.

**Methods**

| Method | Description |
|---|---|
| `now() → float` | Current UNIX timestamp |
| `process(*args, **kwargs)` | Override in subclasses |
| `save_checkpoint(path, extra=None)` | Serialise model weights + Ledger snapshot |
| `load_checkpoint(path) → Dict` | Load weights + metadata |

---

### `LGTVictorOSModule`

```python
class LGTVictorOSModule(VictorOSBaseModule)
```

Pre-built VictorOS module wrapping any `LightweightGravitationalTransformer`.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | — | Pre-constructed LGT model |
| `agent_id` | `str` | `"lgt_core"` | Ledger agent identifier |
| `persist_path` | `str \| None` | `None` | Ledger persistence path |
| `max_force_threshold` | `float` | `40.0` | Mirror Layer containment threshold |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `process(x, return_diagnostics=True)` | `Dict` | Run inference with full VictorOS integration |
| `get_snapshot(x)` | `Dict` | Full attention snapshot for causal tracing |
| `propose_architecture_change(current_config, stability_threshold=0.95)` | `Dict \| None` | Propose structural mutation when stable |

---

## `training`

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

---

### `ContainmentProtocol`

```python
class ContainmentProtocol
```

Per-step safety guard.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ContainmentConfig` | — | Safety configuration |
| `model` | `nn.Module` | — | Model being trained |
| `ledger` | `Ledger \| None` | `None` | Optional event logger |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `step(loss, diagnostics=None)` | `Dict` | Apply all containment checks for one training step |
| `bekenstein_penalty(x)` | `Tensor` | Compute Bekenstein entropy regularisation term |

`step()` return dict keys: `step`, `loss`, `clipped`, `damped`, `stopped`, `stability`, `proposal`.

---

### `MetaCurvatureScheduler`

```python
class MetaCurvatureScheduler
```

Meta-gradient curvature adaptation.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | — | LGT model |
| `lr` | `float` | `0.01` | Meta-learning rate |
| `min_curvature` | `float` | `0.0` | Lower bound |
| `max_curvature` | `float` | `0.5` | Upper bound |

**Methods**

#### `step(val_loss) → Dict[str, float]`

Update curvature parameters based on validation loss delta. Returns `{param_name: new_value}`.

---

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

### `TrainingLoop`

```python
class TrainingLoop
```

Full training orchestrator.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | — | LGT model |
| `optimizer` | `Optimizer` | — | PyTorch optimiser |
| `loss_fn` | `callable` | — | `(logits, targets) → scalar loss` |
| `config` | `TrainingConfig \| None` | `None` | Uses defaults if `None` |
| `containment_config` | `ContainmentConfig \| None` | `None` | Uses defaults if `None` |
| `ledger` | `Ledger \| None` | `None` | Event logger |
| `scheduler` | `LRScheduler \| None` | `None` | LR scheduler |
| `device` | `torch.device \| None` | CPU | Target device |

**Methods**

| Method | Returns | Description |
|---|---|---|
| `train_step(batch, return_diagnostics=False)` | `Dict` | Single training step |
| `eval_step(batch)` | `float` | Single evaluation step; returns val loss |
| `fit(train_iter, val_iter=None, on_proposal=None)` | `Dict` | Full training loop |
| `proposals` (property) | `List[Dict]` | All architecture proposals generated so far |

`fit()` return dict: `{"steps": int, "final_loss": float, "proposals": List[Dict]}`.

---

## `tri_model`

### `CrossGravitationalFusion`

```python
class CrossGravitationalFusion(nn.Module)
```

Cross-gravitational attention fusion for three input streams.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim_model` | `int` | — | Shared stream dimension |
| `num_heads` | `int` | `4` | Cross-attention heads |
| `gravitational_constant` | `float` | `1.0` | Learnable G for mass scaling |
| `dropout` | `float` | `0.1` | Dropout probability |

**Methods**

#### `forward(world, self_, env) → (Tensor, Tensor, Tensor)`

Returns `(world_out, self_out, env_out)`, each `[batch, seq, dim_model]`.

---

### `TriModelTransformer`

```python
class TriModelTransformer(nn.Module)
```

Three-stream world / self / environment cognitive architecture.

**Constructor Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim_model` | `int` | `128` | Shared model dimension |
| `dim_position` | `int` | `64` | Position vector dimension |
| `num_layers` | `int` | `4` | Layers per sub-model |
| `num_heads` | `int` | `4` | Heads per block |
| `vocab_size` | `int \| None` | `None` | Vocabulary size (shared embedding) |
| `max_seq_len` | `int` | `512` | Maximum sequence length per stream |
| `dropout` | `float` | `0.1` | Dropout probability |
| `use_fractal_positions` | `bool` | `False` | Use fractal position embeddings |
| `output_dim` | `int \| None` | `None` | Output projection; defaults to `dim_model` |

**Methods**

#### `forward(world_input, self_input, env_input, return_diagnostics=False, mirror_layer_callback=None) → (Tensor, Dict | None)`

Returns `(output [batch, seq, output_dim], diagnostics)`.

#### `get_tri_snapshot(world_input, self_input, env_input) → Dict`

Returns per-stream snapshots and fusion diagnostics.

---

## `export_edge_model`

### `PRESETS`

```python
PRESETS: Dict[str, Dict[str, Any]] = {
    "edge_150k":  {"dim_model": 64,  "dim_position": 32,  "num_layers": 2, "num_heads": 2,  "curvature": 0.10},
    "meta_probe": {"dim_model": 128, "dim_position": 64,  "num_layers": 4, "num_heads": 4,  "curvature": 0.15},
    "fractal_res":{"dim_model": 256, "dim_position": 128, "num_layers": 6, "num_heads": 8,  "curvature": 0.25},
    "victorcos":  {"dim_model": 192, "dim_position": 96,  "num_layers": 5, "num_heads": 6,  "curvature": 0.18},
}
```

---

### `build_model`

```python
def build_model(
    config_name: str = "edge_150k",
    vocab_size: int = 32000,
    max_seq_len: int = 512,
    use_fractal_positions: bool = False,
    **kwargs,
) -> LightweightGravitationalTransformer
```

Build a model from a named preset with optional overrides.

---

### `export_torchscript`

```python
def export_torchscript(
    model: nn.Module,
    example_input: Tensor,
    output_path: str,
) -> str
```

Trace model with TorchScript and save. Returns the saved path.

---

### `quantize_dynamic`

```python
def quantize_dynamic(
    model: nn.Module,
    dtype: str = "int8",
) -> nn.Module
```

Apply dynamic quantisation to `nn.Linear` layers. `dtype` must be `"int8"` or `"float16"`.

---

### `save_checkpoint`

```python
def save_checkpoint(
    model: nn.Module,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str
```

Save model weights + metadata. Returns the saved path.

---

### `export_edge_model` (function)

```python
def export_edge_model(
    config_name: str = "edge_150k",
    vocab_size: int = 32000,
    max_seq_len: int = 512,
    quantize: str = "none",
    output_dir: str = "exported_models",
    use_fractal_positions: bool = False,
    example_seq_len: int = 64,
) -> Dict[str, str]
```

Full export pipeline: build → quantise → TorchScript → save.

Returns `{"checkpoint": str, "torchscript": str, "config": Dict}`.
