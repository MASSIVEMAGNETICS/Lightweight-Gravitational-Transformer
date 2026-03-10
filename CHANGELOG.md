# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] – 2024-01-01

### Added
- `GravitationalAttentionHead` – single-head gravitational attention using Newton's law
- `MultiHeadGravitationalAttention` – multi-head extension with independent per-head `G` parameters and `get_attention_diagnostics()`
- `FractalPositionEmbedding` – multi-scale position encoding with power-law (fractal) frequency spectrum and learnable residuals
- `CurvedPositionEmbedding` – learnable positional vectors on a curved manifold
- `LightweightGravitationalBlock` – single transformer block: gravitational attention + lightweight FFN (2× expansion) + layer norms
- `LightweightGravitationalTransformer` – full transformer stack with optional vocabulary embedding, tied weights, Mirror Layer callbacks, and attention snapshots
- `Ledger` – append-only JSONL event log with in-memory buffering and file-persistence
- `MirrorLayer` – real-time introspection hook with rolling stability scoring and correction callbacks
- `@victoros_module` – class decorator for packaging LGT agents as VictorOS cognitive modules
- `VictorOSBaseModule` – base class providing `Ledger`, `MirrorLayer`, `save_checkpoint`, and `load_checkpoint`
- `LGTVictorOSModule` – concrete VictorOS module wrapping any `LightweightGravitationalTransformer`
- `ContainmentProtocol` – per-step safety guard (gradient clipping, force dampening, Bekenstein entropy penalty, divergence detection, architecture proposals)
- `MetaCurvatureScheduler` – meta-gradient curvature adaptation driven by validation loss
- `TrainingLoop` – full training orchestrator integrating all physics-aware constraints
- `CrossGravitationalFusion` – gravitational cross-attention for tri-model stream fusion
- `TriModelTransformer` – world / self / environment three-stream cognitive architecture
- `export_edge_model.py` – TorchScript export and dynamic INT8 / FP16 quantisation with four size presets (`edge_150k`, `meta_probe`, `fractal_res`, `victorcos`)
- `benchmarks/benchmark_lgt.py` – comprehensive performance benchmarking suite
- `tests/test_lgt.py` – 60+ pytest test cases covering all components
- `pyproject.toml` – package metadata and build configuration
- `LICENSE` – MIT licence
- `CONTRIBUTING.md` – contributor guidelines
- `CHANGELOG.md` – this file
- `docs/` – enterprise documentation (installation guide, user guide, API reference, architecture deep-dive)
- `examples/` – five runnable example scripts

[Unreleased]: https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer/releases/tag/v0.1.0
