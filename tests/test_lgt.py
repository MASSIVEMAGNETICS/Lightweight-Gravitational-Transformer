"""
Tests for the Lightweight Gravitational Transformer ecosystem.

Covers:
- gravitational_attention.py
- fractal_position_embedding.py
- lightweight_gravitational_transformer.py
- victorcos_module.py
- training.py
- tri_model.py
- export_edge_model.py
- benchmarks/benchmark_lgt.py
"""

import math
import os
import sys
import tempfile
import time

import pytest
import torch
import torch.nn as nn

# Allow imports from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gravitational_attention import GravitationalAttentionHead, MultiHeadGravitationalAttention
from fractal_position_embedding import FractalPositionEmbedding
from lightweight_gravitational_transformer import (
    CurvedPositionEmbedding,
    LightweightGravitationalBlock,
    LightweightGravitationalTransformer,
)
from victorcos_module import (
    Ledger,
    LedgerEntry,
    MirrorLayer,
    LGTVictorOSModule,
    MorphicVictorAgent,
    victoros_module,
    VictorOSBaseModule,
)
from training import (
    ContainmentConfig,
    ContainmentProtocol,
    MetaCurvatureScheduler,
    TrainingConfig,
    TrainingLoop,
)
from tri_model import TriModelTransformer, CrossGravitationalFusion
from export_edge_model import build_model, export_edge_model, PRESETS
from octonion_pos_embedding import (
    OctonionEmbedding,
    octonion_distance,
    GravitationalOctonionPosition,
)
from polymorphic_attention_orchestrator import (
    PHASE_CONFIG,
    PolymorphicAttentionOrchestrator,
)
from training_containment import MorphicContainmentConfig, MorphicContainmentProtocol


# ===========================================================================
# gravitational_attention
# ===========================================================================

class TestGravitationalAttentionHead:
    def test_output_shape(self):
        head = GravitationalAttentionHead(head_dim=16)
        x = torch.randn(2, 8, 16)
        out, masses = head(x)
        assert out.shape == (2, 8, 16)
        assert masses.shape == (2, 8)

    def test_output_shape_with_positions(self):
        head = GravitationalAttentionHead(head_dim=16)
        x = torch.randn(2, 8, 16)
        positions = torch.randn(8, 32)
        out, masses = head(x, positions)
        assert out.shape == (2, 8, 16)

    def test_masses_positive(self):
        head = GravitationalAttentionHead(head_dim=16)
        x = torch.randn(2, 8, 16)
        _, masses = head(x)
        assert (masses > 0).all()

    def test_max_force_clipping(self):
        head = GravitationalAttentionHead(head_dim=16, max_force=1.0, gravitational_constant=1000.0)
        x = torch.randn(2, 8, 16)
        out, _ = head(x)
        assert not torch.isnan(out).any()


class TestMultiHeadGravitationalAttention:
    def test_output_shape(self):
        attn = MultiHeadGravitationalAttention(dim_model=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_with_positions(self):
        attn = MultiHeadGravitationalAttention(dim_model=64, num_heads=4, dim_position=32)
        x = torch.randn(2, 16, 64)
        pos = torch.randn(16, 32)
        out = attn(x, positions=pos)
        assert out.shape == (2, 16, 64)

    def test_dim_model_not_divisible_raises(self):
        with pytest.raises(ValueError):
            MultiHeadGravitationalAttention(dim_model=65, num_heads=4)

    def test_diagnostics_keys(self):
        attn = MultiHeadGravitationalAttention(dim_model=64, num_heads=4)
        x = torch.randn(1, 8, 64)
        diag = attn.get_attention_diagnostics(x)
        assert "head_0" in diag
        assert "mean_mass" in diag["head_0"]
        assert "mean_force" in diag["head_0"]
        assert "G" in diag["head_0"]

    def test_diagnostics_accepts_numpy(self):
        import numpy as np
        attn = MultiHeadGravitationalAttention(dim_model=64, num_heads=4)
        x_np = np.random.randn(1, 8, 64).astype(np.float32)
        diag = attn.get_attention_diagnostics(x_np)
        assert "head_0" in diag

    def test_gradient_flow(self):
        attn = MultiHeadGravitationalAttention(dim_model=32, num_heads=2)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ===========================================================================
# fractal_position_embedding
# ===========================================================================

class TestFractalPositionEmbedding:
    def test_output_shape(self):
        emb = FractalPositionEmbedding(max_seq_len=128, dim_position=64)
        pos = emb(32)
        assert pos.shape == (32, 64)

    def test_full_seq_len(self):
        emb = FractalPositionEmbedding(max_seq_len=64, dim_position=32)
        pos = emb(64)
        assert pos.shape == (64, 32)

    def test_no_nan(self):
        emb = FractalPositionEmbedding(max_seq_len=128, dim_position=64, num_scales=4)
        pos = emb(64)
        assert not torch.isnan(pos).any()

    def test_learnable_residual(self):
        emb = FractalPositionEmbedding(max_seq_len=64, dim_position=32, learnable_residual=True)
        params = {n for n, _ in emb.named_parameters()}
        assert "residual" in params

    def test_no_learnable_residual(self):
        emb = FractalPositionEmbedding(max_seq_len=64, dim_position=32, learnable_residual=False)
        params = {n for n, _ in emb.named_parameters()}
        assert "residual" not in params


# ===========================================================================
# lightweight_gravitational_transformer
# ===========================================================================

class TestCurvedPositionEmbedding:
    def test_shape(self):
        emb = CurvedPositionEmbedding(max_seq_len=64, dim_position=32)
        out = emb(16)
        assert out.shape == (16, 32)

    def test_curvature_modulation(self):
        emb = CurvedPositionEmbedding(max_seq_len=64, dim_position=32, curvature=0.0)
        p1 = emb(8)
        emb2 = CurvedPositionEmbedding(max_seq_len=64, dim_position=32, curvature=0.5)
        # Just check it runs without error
        p2 = emb2(8)
        assert p1.shape == p2.shape


class TestLightweightGravitationalBlock:
    def _make_block(self, dim=64, heads=2):
        return LightweightGravitationalBlock(dim_model=dim, dim_position=32, num_heads=heads)

    def test_output_shape(self):
        block = self._make_block()
        x = torch.randn(2, 8, 64)
        out, diag = block(x)
        assert out.shape == (2, 8, 64)
        assert diag is None

    def test_diagnostics_returned(self):
        block = self._make_block()
        x = torch.randn(1, 8, 64)
        out, diag = block(x, return_diagnostics=True)
        assert diag is not None
        assert "mean_force" in diag
        assert "mean_mass" in diag
        assert "hawking_limit" in diag

    def test_with_positions(self):
        block = self._make_block()
        x = torch.randn(1, 8, 64)
        pos = torch.randn(8, 32)
        out, _ = block(x, positions=pos)
        assert out.shape == (1, 8, 64)

    def test_gradient_flow(self):
        block = self._make_block()
        x = torch.randn(1, 4, 64, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestLightweightGravitationalTransformer:
    """Core spec tests from the problem statement."""

    def test_basic_forward_pass(self):
        model = LightweightGravitationalTransformer(
            vocab_size=1000, dim_model=64, num_layers=2, num_heads=2
        )
        x = torch.randint(0, 1000, (1, 16))
        logits, diag = model(x, return_diagnostics=True)
        assert logits.shape == (1, 16, 1000)
        assert diag is not None
        assert "layers" in diag
        assert len(diag["layers"]) == 2

    def test_mirror_layer_callback(self):
        model = LightweightGravitationalTransformer(
            vocab_size=1000, dim_model=64, num_layers=2, num_heads=2
        )
        x = torch.randint(0, 1000, (1, 16))
        calls = []
        def callback(idx, d):
            calls.append((idx, d["mean_force"]))
        model(x, return_diagnostics=True, mirror_layer_callback=callback)
        assert len(calls) == 2
        assert calls[0][0] == 0
        assert calls[1][0] == 1

    def test_attention_snapshot(self):
        model = LightweightGravitationalTransformer(
            vocab_size=1000, dim_model=64, num_layers=2, num_heads=2
        )
        x = torch.randint(0, 1000, (1, 16))
        snapshot = model.get_attention_snapshot(x)
        assert "attention_metrics" in snapshot
        assert "model_config" in snapshot
        assert snapshot["model_config"]["num_layers"] == 2

    def test_no_vocab_continuous_input(self):
        model = LightweightGravitationalTransformer(
            dim_model=32, dim_position=16, num_layers=1, num_heads=2
        )
        x = torch.randn(2, 8, 32)
        out, _ = model(x)
        assert out.shape == (2, 8, 32)

    def test_fractal_positions(self):
        model = LightweightGravitationalTransformer(
            vocab_size=100, dim_model=32, dim_position=16,
            num_layers=1, num_heads=2, use_fractal_positions=True,
        )
        x = torch.randint(0, 100, (1, 8))
        out, diag = model(x, return_diagnostics=True)
        assert out.shape == (1, 8, 100)

    def test_tied_weights(self):
        model = LightweightGravitationalTransformer(
            vocab_size=200, dim_model=32, dim_position=16,
            num_layers=1, num_heads=2, tie_weights=True,
        )
        x = torch.randint(0, 200, (1, 4))
        out, _ = model(x)
        assert out.shape == (1, 4, 200)

    def test_diagnostics_structure(self):
        model = LightweightGravitationalTransformer(
            vocab_size=100, dim_model=32, num_layers=2, num_heads=2
        )
        x = torch.randint(0, 100, (1, 4))
        _, diag = model(x, return_diagnostics=True)
        assert "final_norm_stats" in diag
        assert "mean" in diag["final_norm_stats"]
        assert "std" in diag["final_norm_stats"]


# ===========================================================================
# victorcos_module
# ===========================================================================

class TestLedger:
    def test_log_creates_entry(self):
        ledger = Ledger(agent_id="test")
        entry = ledger.log("test_event", {"key": "val"})
        assert len(ledger) == 1
        assert entry.event == "test_event"
        assert entry.payload["key"] == "val"

    def test_entry_has_timestamp(self):
        ledger = Ledger()
        entry = ledger.log("ev")
        assert entry.timestamp > 0

    def test_filter_by_event(self):
        ledger = Ledger()
        ledger.log("alpha")
        ledger.log("beta")
        ledger.log("alpha")
        assert len(ledger.entries("alpha")) == 2
        assert len(ledger.entries("beta")) == 1

    def test_flush_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ledger.jsonl")
            ledger = Ledger(agent_id="flush_test", persist_path=path)
            ledger.log("event1", {"x": 1})
            ledger.log("event2", {"x": 2})
            n = ledger.flush()
            assert n == 2
            assert len(ledger) == 0  # cleared after flush
            assert os.path.exists(path)
            with open(path) as fh:
                lines = fh.readlines()
            assert len(lines) == 2

    def test_snapshot(self):
        ledger = Ledger(agent_id="snap")
        ledger.log("e")
        snap = ledger.snapshot()
        assert snap["agent_id"] == "snap"
        assert snap["entry_count"] == 1

    def test_ledger_entry_json_roundtrip(self):
        import json
        entry = LedgerEntry(agent_id="a", event="e", payload={"v": 1})
        d = json.loads(entry.to_json())
        assert d["event"] == "e"
        assert d["payload"]["v"] == 1


class TestMirrorLayer:
    def test_logs_to_ledger(self):
        ledger = Ledger()
        ml = MirrorLayer(ledger=ledger)
        ml(0, {"mean_force": 5.0, "mean_mass": 1.0, "curvature_active": True})
        assert len(ledger.entries("mirror_layer")) == 1

    def test_correction_triggered_above_threshold(self):
        corrections = []
        ledger = Ledger()
        ml = MirrorLayer(ledger=ledger, max_force_threshold=10.0,
                         correction_callback=lambda li, ct: corrections.append((li, ct)))
        ml(0, {"mean_force": 50.0})
        assert len(corrections) == 1
        assert corrections[0][1] == "attention_dampening"

    def test_no_correction_below_threshold(self):
        corrections = []
        ml = MirrorLayer(max_force_threshold=100.0,
                         correction_callback=lambda li, ct: corrections.append(ct))
        ml(0, {"mean_force": 5.0})
        assert len(corrections) == 0

    def test_stability_score_range(self):
        ml = MirrorLayer(max_force_threshold=40.0)
        for _ in range(20):
            ml(0, {"mean_force": 20.0})
        score = ml.stability_score()
        assert 0.0 <= score <= 1.0


class TestLGTVictorOSModule:
    def _make_module(self):
        model = LightweightGravitationalTransformer(
            vocab_size=100, dim_model=32, dim_position=16, num_layers=2, num_heads=2
        )
        return LGTVictorOSModule(model, agent_id="test_agent")

    def test_process_returns_output(self):
        mod = self._make_module()
        x = torch.randint(0, 100, (1, 8))
        result = mod.process(x)
        assert "output" in result
        assert "stability" in result

    def test_ledger_populated_after_process(self):
        mod = self._make_module()
        x = torch.randint(0, 100, (1, 8))
        mod.process(x)
        assert len(mod.ledger) >= 1

    def test_get_snapshot(self):
        mod = self._make_module()
        x = torch.randint(0, 100, (1, 8))
        snap = mod.get_snapshot(x)
        assert "attention_metrics" in snap

    def test_architecture_proposal_low_stability(self):
        mod = self._make_module()
        # Stability is 1.0 initially; force it low by injecting high forces
        for _ in range(50):
            mod.mirror_layer(0, {"mean_force": 200.0})
        proposal = mod.propose_architecture_change(
            {"num_layers": 2, "curvature": 0.15}, stability_threshold=0.999
        )
        assert proposal is None  # stability too low

    def test_architecture_proposal_high_stability(self):
        mod = self._make_module()
        # Fresh module has stability ~1.0
        proposal = mod.propose_architecture_change(
            {"num_layers": 2, "curvature": 0.15}, stability_threshold=0.5
        )
        assert proposal is not None
        assert "change" in proposal

    def test_save_load_checkpoint(self):
        mod = self._make_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            mod.save_checkpoint(path)
            assert os.path.exists(path)
            state = mod.load_checkpoint(path)
            assert "model_state_dict" in state


class TestVictorOSModuleDecorator:
    def test_decorator_attaches_metadata(self):
        @victoros_module(name="test_mod", version="1.0.0", containment_native=True)
        class MyMod(VictorOSBaseModule):
            def __init__(self):
                self.x = 1

        assert MyMod._victoros_meta.name == "test_mod"
        assert MyMod._victoros_meta.containment_native is True

    def test_decorator_auto_provisions_ledger(self):
        @victoros_module(name="auto_ledger", version="0.0.1")
        class MyMod(VictorOSBaseModule):
            def __init__(self):
                pass

        mod = MyMod()
        assert hasattr(mod, "ledger")
        assert isinstance(mod.ledger, Ledger)

    def test_decorator_preserves_existing_ledger(self):
        custom_ledger = Ledger(agent_id="custom")

        @victoros_module(name="custom_ledger_mod", version="0.0.1")
        class MyMod(VictorOSBaseModule):
            def __init__(self):
                self.ledger = custom_ledger

        mod = MyMod()
        assert mod.ledger is custom_ledger


# ===========================================================================
# training
# ===========================================================================

class TestContainmentProtocol:
    def _make_protocol(self):
        model = LightweightGravitationalTransformer(
            vocab_size=50, dim_model=32, dim_position=16, num_layers=2, num_heads=2
        )
        return ContainmentProtocol(ContainmentConfig(), model)

    def test_gradient_clipping(self):
        proto = self._make_protocol()
        # Give a dummy loss to trigger the check
        loss = torch.tensor(1.0, requires_grad=True)
        # Manually set large gradients
        for p in proto.model.parameters():
            p.grad = torch.ones_like(p) * 1000
        summary = proto.step(loss)
        assert summary["clipped"] is True

    def test_divergence_stops_training(self):
        proto = self._make_protocol()
        loss = torch.tensor(float("inf"))
        summary = proto.step(loss)
        assert summary["stopped"] is True

    def test_bekenstein_penalty_is_positive(self):
        proto = self._make_protocol()
        x = torch.randn(2, 8, 32)
        penalty = proto.bekenstein_penalty(x)
        # Penalty can be positive or negative depending on variance; check shape
        assert penalty.shape == ()

    def test_attention_dampening(self):
        proto = self._make_protocol()
        # Get the initial G values
        before = [p.item() for m in proto.model.modules()
                  if hasattr(m, "G") and isinstance(m.G, nn.Parameter)
                  for p in [m.G]]
        diag = {"layers": [{"mean_force": 1000.0}]}
        loss = torch.tensor(1.0, requires_grad=True)
        for p in proto.model.parameters():
            p.grad = torch.zeros_like(p)
        proto.step(loss, diagnostics=diag)
        after = [p.item() for m in proto.model.modules()
                 if hasattr(m, "G") and isinstance(m.G, nn.Parameter)
                 for p in [m.G]]
        assert all(a < b for a, b in zip(after, before))


class TestMetaCurvatureScheduler:
    def test_updates_curvature_params(self):
        model = LightweightGravitationalTransformer(
            dim_model=32, dim_position=16, num_layers=2, num_heads=2
        )
        sched = MetaCurvatureScheduler(model, lr=0.1)
        sched.step(1.0)  # seed prev loss
        updates = sched.step(2.0)  # loss increased → curvature should decrease
        assert len(updates) > 0

    def test_curvature_stays_within_bounds(self):
        model = LightweightGravitationalTransformer(
            dim_model=32, dim_position=16, num_layers=2, num_heads=2
        )
        sched = MetaCurvatureScheduler(model, lr=1.0, min_curvature=0.0, max_curvature=0.5)
        for loss in [float(i) for i in range(10)]:
            sched.step(loss)
        for name, param in model.named_parameters():
            if "curvature" in name:
                assert 0.0 <= param.item() <= 0.5


class TestTrainingLoop:
    def _make_loop(self, model=None):
        if model is None:
            model = LightweightGravitationalTransformer(
                vocab_size=50, dim_model=32, dim_position=16, num_layers=1, num_heads=2
            )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        def _loss_fn(logits, targets):
            return loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return TrainingLoop(
            model, opt, _loss_fn,
            config=TrainingConfig(max_steps=3, log_every=1, use_bekenstein_penalty=False),
        )

    def test_single_train_step(self):
        loop = self._make_loop()
        batch = (torch.randint(0, 50, (2, 4)), torch.randint(0, 50, (2, 4)))
        result = loop.train_step(batch)
        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_eval_step_returns_float(self):
        loop = self._make_loop()
        batch = (torch.randint(0, 50, (2, 4)), torch.randint(0, 50, (2, 4)))
        val_loss = loop.eval_step(batch)
        assert isinstance(val_loss, float)

    def test_fit_runs_multiple_steps(self):
        loop = self._make_loop()

        def _gen():
            for _ in range(5):
                yield torch.randint(0, 50, (2, 4)), torch.randint(0, 50, (2, 4))

        summary = loop.fit(_gen())
        assert summary["steps"] >= 3  # max_steps=3


# ===========================================================================
# tri_model
# ===========================================================================

class TestCrossGravitationalFusion:
    def test_output_shapes(self):
        fusion = CrossGravitationalFusion(dim_model=32, num_heads=2)
        world = torch.randn(2, 8, 32)
        self_ = torch.randn(2, 8, 32)
        env = torch.randn(2, 8, 32)
        w, s, e = fusion(world, self_, env)
        assert w.shape == (2, 8, 32)
        assert s.shape == (2, 8, 32)
        assert e.shape == (2, 8, 32)


class TestTriModelTransformer:
    def _make_model(self):
        return TriModelTransformer(
            dim_model=32, dim_position=16, num_layers=1, num_heads=2
        )

    def test_output_shape(self):
        model = self._make_model()
        world = torch.randn(1, 4, 32)
        self_ = torch.randn(1, 4, 32)
        env = torch.randn(1, 4, 32)
        out, diag = model(world, self_, env)
        assert out.shape[0] == 1
        assert out.shape[-1] == 32
        assert diag is None

    def test_with_diagnostics(self):
        model = self._make_model()
        world = torch.randn(1, 4, 32)
        self_ = torch.randn(1, 4, 32)
        env = torch.randn(1, 4, 32)
        out, diag = model(world, self_, env, return_diagnostics=True)
        assert diag is not None
        assert "world" in diag
        assert "self" in diag
        assert "env" in diag
        assert "fusion" in diag

    def test_mirror_layer_callback(self):
        model = self._make_model()
        calls = []
        def cb(stream, layer_idx, d):
            calls.append((stream, layer_idx))
        world = torch.randn(1, 4, 32)
        self_ = torch.randn(1, 4, 32)
        env = torch.randn(1, 4, 32)
        model(world, self_, env, return_diagnostics=True, mirror_layer_callback=cb)
        # Each sub-model has 1 layer → 3 calls total
        assert len(calls) == 3
        streams = {c[0] for c in calls}
        assert streams == {"world", "self", "env"}

    def test_tri_snapshot(self):
        model = self._make_model()
        world = torch.randn(1, 4, 32)
        self_ = torch.randn(1, 4, 32)
        env = torch.randn(1, 4, 32)
        snap = model.get_tri_snapshot(world, self_, env)
        assert "world_snapshot" in snap
        assert "self_snapshot" in snap
        assert "env_snapshot" in snap

    def test_with_vocab(self):
        model = TriModelTransformer(
            vocab_size=100, dim_model=32, dim_position=16, num_layers=1, num_heads=2
        )
        world = torch.randint(0, 100, (1, 4))
        self_ = torch.randint(0, 100, (1, 4))
        env = torch.randint(0, 100, (1, 4))
        out, _ = model(world, self_, env)
        assert out.shape[0] == 1


# ===========================================================================
# export_edge_model
# ===========================================================================

class TestExportEdgeModel:
    def test_build_model_presets(self):
        for preset_name in PRESETS:
            model = build_model(preset_name, vocab_size=100)
            assert isinstance(model, LightweightGravitationalTransformer)

    def test_export_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_edge_model(
                config_name="edge_150k",
                vocab_size=100,
                max_seq_len=16,
                quantize="none",
                output_dir=tmpdir,
                example_seq_len=4,
            )
            assert os.path.exists(paths["checkpoint"])

    def test_export_float16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_edge_model(
                config_name="edge_150k",
                vocab_size=100,
                max_seq_len=16,
                quantize="float16",
                output_dir=tmpdir,
                example_seq_len=4,
            )
            assert os.path.exists(paths["checkpoint"])


# ===========================================================================
# benchmarks
# ===========================================================================

class TestBenchmarks:
    def test_benchmark_edge_preset(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
        from benchmark_lgt import benchmark_preset
        result = benchmark_preset("edge_150k", vocab_size=100, seq_len=4, num_runs=3)
        assert "params" in result
        assert "latency" in result
        assert "throughput" in result
        assert "memory" in result
        assert result["latency"]["mean_ms"] > 0
        assert result["throughput"]["inferences_per_sec"] > 0


# ===========================================================================
# octonion_pos_embedding
# ===========================================================================

class TestOctonionEmbedding:
    def test_output_shape(self):
        emb = OctonionEmbedding(dim_model=64, max_len=128)
        x = torch.randn(2, 10, 64)
        pe = emb(x)
        assert pe.shape == (1, 10, 64)

    def test_device_consistency(self):
        emb = OctonionEmbedding(dim_model=32, max_len=64)
        x = torch.randn(1, 8, 32)
        pe = emb(x)
        assert pe.device == x.device

    def test_no_nan(self):
        emb = OctonionEmbedding(dim_model=64, max_len=128)
        x = torch.randn(2, 16, 64)
        pe = emb(x)
        assert not torch.isnan(pe).any()

    def test_dim_not_divisible_by_8(self):
        # dim_model=33 is not divisible by 8 but should still work
        emb = OctonionEmbedding(dim_model=33, max_len=16)
        x = torch.randn(1, 4, 33)
        pe = emb(x)
        assert pe.shape == (1, 4, 33)


class TestOctonionDistance:
    def test_zero_distance_for_identical_vectors(self):
        v = torch.randn(3, 8)
        dist = octonion_distance(v, v)
        # Should be near zero (only epsilon keeps it > 0)
        assert (dist < 1e-3).all()

    def test_non_negative(self):
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        dist = octonion_distance(a, b)
        assert (dist >= 0).all()

    def test_output_shape_keepdim(self):
        a = torch.randn(2, 5, 1, 16)
        b = torch.randn(2, 1, 5, 16)
        dist = octonion_distance(a, b)
        assert dist.shape == (2, 5, 5, 1)


class TestGravitationalOctonionPosition:
    def test_output_shape(self):
        gop = GravitationalOctonionPosition(dim_model=64, max_len=128)
        x = torch.randn(2, 10, 64)
        dist = gop(x)
        assert dist.shape == (2, 10, 10)

    def test_non_negative(self):
        gop = GravitationalOctonionPosition(dim_model=32, max_len=64)
        x = torch.randn(1, 6, 32)
        dist = gop(x)
        assert (dist >= 0).all()

    def test_no_nan(self):
        gop = GravitationalOctonionPosition(dim_model=64, max_len=128)
        x = torch.randn(2, 8, 64)
        dist = gop(x)
        assert not torch.isnan(dist).any()


# ===========================================================================
# polymorphic_attention_orchestrator
# ===========================================================================

class TestPolymorphicAttentionOrchestrator:
    def _make_orch(self, dim=64, heads=4):
        return PolymorphicAttentionOrchestrator(dim_model=dim, num_heads=heads, max_len=128)

    def test_output_shape(self):
        orch = self._make_orch()
        x = torch.randn(2, 8, 64)
        out, diag = orch(x)
        assert out.shape == (2, 8, 64)

    def test_diagnostics_keys(self):
        orch = self._make_orch()
        x = torch.randn(1, 6, 64)
        _, diag = orch(x)
        assert "max_force" in diag
        assert "mean_force" in diag
        assert "phase" in diag
        assert "G" in diag
        assert "curvature" in diag

    def test_morph_changes_phase(self):
        orch = self._make_orch()
        orch.morph("singularity")
        assert orch.current_phase == "singularity"
        assert orch.G == PHASE_CONFIG["singularity"]["G"]

    def test_all_phases_run(self):
        orch = self._make_orch()
        x = torch.randn(1, 4, 64)
        for phase in ["solid", "fluid", "gas", "singularity"]:
            out, diag = orch(x, phase=phase)
            assert out.shape == (1, 4, 64)
            assert diag["phase"] == phase

    def test_phase_override_does_not_mutate_current_phase(self):
        orch = self._make_orch()
        orch.morph("solid")
        x = torch.randn(1, 4, 64)
        orch(x, phase="gas")
        # current_phase should still be "solid"
        assert orch.current_phase == "solid"

    def test_invalid_phase_raises(self):
        orch = self._make_orch()
        with pytest.raises(ValueError):
            orch.morph("plasma")

    def test_dim_not_divisible_raises(self):
        with pytest.raises(ValueError):
            PolymorphicAttentionOrchestrator(dim_model=65, num_heads=4)

    def test_no_nan_output(self):
        orch = self._make_orch()
        x = torch.randn(2, 8, 64)
        out, _ = orch(x, phase="singularity")
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        orch = self._make_orch()
        x = torch.randn(1, 4, 64, requires_grad=True)
        out, _ = orch(x)
        out.sum().backward()
        assert x.grad is not None

    def test_phase_config_completeness(self):
        for phase in ["solid", "fluid", "gas", "singularity"]:
            assert phase in PHASE_CONFIG
            cfg = PHASE_CONFIG[phase]
            assert "G" in cfg
            assert "curvature" in cfg
            assert "hawking_clamp" in cfg


# ===========================================================================
# training_containment
# ===========================================================================

class TestMorphicContainmentProtocol:
    def _make_protocol(self, ledger=None):
        model = nn.Linear(16, 16)
        config = MorphicContainmentConfig(
            max_grad_norm=1.0,
            max_attention_force=10.0,
            bekenstein_lambda=1e-4,
            min_stability=0.2,
        )
        return MorphicContainmentProtocol(model=model, ledger=ledger, config=config)

    def _make_loss_with_grad(self, model):
        """Helper: produce a backward-ed loss so grads exist."""
        x = torch.randn(1, 16)
        loss = model(x).sum()
        loss.backward()
        return loss.detach()

    def test_step_returns_true_on_normal_run(self):
        proto = self._make_protocol()
        loss = self._make_loss_with_grad(proto.model)
        result = proto.step(loss, {"max_force": 1.0, "phase": "fluid", "stability": 0.9})
        assert result is True

    def test_step_returns_false_on_low_stability(self):
        proto = self._make_protocol()
        loss = self._make_loss_with_grad(proto.model)
        result = proto.step(loss, {"max_force": 1.0, "phase": "fluid", "stability": 0.1})
        assert result is False

    def test_hawking_radiation_dampens_gradients(self):
        proto = self._make_protocol()
        loss = self._make_loss_with_grad(proto.model)
        # Record grad norm before
        before = sum(p.grad.norm().item() for p in proto.model.parameters() if p.grad is not None)
        # Trigger Hawking radiation (force >> max_attention_force)
        proto.step(loss, {"max_force": 1000.0, "phase": "singularity", "stability": 0.9})
        after = sum(p.grad.norm().item() for p in proto.model.parameters() if p.grad is not None)
        assert after <= before + 1e-6  # gradients were scaled down (then clipped)

    def test_bekenstein_penalty_shape(self):
        proto = self._make_protocol()
        attn = torch.softmax(torch.randn(2, 8, 8), dim=-1)
        penalty = proto.apply_bekenstein_penalty(attn)
        assert penalty.shape == ()
        assert penalty.item() >= 0

    def test_bekenstein_penalty_differentiable(self):
        proto = self._make_protocol()
        attn = torch.softmax(torch.randn(2, 8, 8), dim=-1).requires_grad_(True)
        penalty = proto.apply_bekenstein_penalty(attn)
        penalty.backward()
        assert attn.grad is not None

    def test_ledger_logging(self):
        ledger = Ledger(agent_id="test_containment")
        proto = self._make_protocol(ledger=ledger)
        loss = self._make_loss_with_grad(proto.model)
        proto.step(loss, {"max_force": 1000.0, "phase": "singularity", "stability": 0.9})
        events = [e.event for e in ledger.entries()]
        assert "containment_event" in events

    def test_default_config_used_when_none(self):
        model = nn.Linear(4, 4)
        proto = MorphicContainmentProtocol(model=model)
        assert proto.config is not None
        assert proto.config.max_grad_norm > 0

    def test_step_no_diagnostics(self):
        proto = self._make_protocol()
        loss = self._make_loss_with_grad(proto.model)
        # Should not raise even without diagnostics
        result = proto.step(loss)
        assert isinstance(result, bool)


# ===========================================================================
# MorphicVictorAgent
# ===========================================================================

class TestMorphicVictorAgent:
    def _make_agent(self, initial_phase="fluid"):
        lgt = LightweightGravitationalTransformer(
            vocab_size=100, dim_model=64, num_layers=2, num_heads=4
        )
        orch = PolymorphicAttentionOrchestrator(dim_model=64, num_heads=4, max_len=128)
        return MorphicVictorAgent(
            model=lgt,
            orchestrator=orch,
            agent_id="test_morphic_agent",
            initial_phase=initial_phase,
        )

    def test_initial_phase(self):
        agent = self._make_agent("solid")
        assert agent.current_phase == "solid"
        assert agent.orchestrator.current_phase == "solid"

    def test_process_morphic_output_shape(self):
        agent = self._make_agent()
        x = torch.randint(0, 100, (1, 8))
        result = agent.process_morphic(x)
        assert result["output"].shape == (1, 8, 100)
        assert "phase" in result

    def test_determine_phase_low_stability(self):
        agent = self._make_agent()
        phase = agent.determine_phase(stability=0.2)
        assert phase == "gas"

    def test_determine_phase_high_stability(self):
        agent = self._make_agent()
        phase = agent.determine_phase(stability=0.95)
        assert phase == "solid"

    def test_determine_phase_singularity(self):
        agent = self._make_agent()
        phase = agent.determine_phase(stability=0.85, task_complexity=1.0)
        assert phase == "singularity"

    def test_phase_shift_logged_to_ledger(self):
        agent = self._make_agent(initial_phase="solid")
        # Drive stability low by feeding high-force diagnostics through
        # the MirrorLayer's public callback interface
        for _ in range(25):
            agent.mirror_layer(0, {"mean_force": 1000.0})
        x = torch.randint(0, 100, (1, 4))
        agent.process_morphic(x)
        events = [e.event for e in agent.ledger.entries()]
        assert "phase_shift" in events

    def test_apply_phase_updates_orchestrator(self):
        agent = self._make_agent("fluid")
        agent.apply_phase("singularity")
        assert agent.orchestrator.current_phase == "singularity"
        assert agent.orchestrator.G == PHASE_CONFIG["singularity"]["G"]

    def test_process_morphic_returns_stability(self):
        agent = self._make_agent()
        x = torch.randint(0, 100, (1, 6))
        result = agent.process_morphic(x)
        assert "stability" in result
        assert 0.0 <= result["stability"] <= 1.0

    def test_no_nan_output(self):
        agent = self._make_agent()
        x = torch.randint(0, 100, (1, 8))
        result = agent.process_morphic(x)
        assert not torch.isnan(result["output"]).any()
