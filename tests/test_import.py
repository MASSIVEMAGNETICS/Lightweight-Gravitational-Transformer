"""
Tests for the boxol_flower package.

Validates importability, geometry correctness, voxel counts, simulation
methods, and GuiController error handling — all in a headless environment.

Voxel counts for default parameters (layers=6, spacing=3.5, z_modulo=3):
    sacred  =   5   (fixed 5-point cross)
    petal   = 126   (hexagonal rings 1-6)
    total   = 131
"""

from __future__ import annotations

import numpy as np
import pytest

from boxol_flower import BoxolFlower, GuiController, __version__
from boxol_flower.boxol import _build_grid, _SACRED_COORDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42


def make_flower(**kwargs) -> BoxolFlower:
    """Return a BoxolFlower with a fixed seed and caller-supplied overrides."""
    kwargs.setdefault("seed", SEED)
    return BoxolFlower(**kwargs)


# ---------------------------------------------------------------------------
# Package smoke tests
# ---------------------------------------------------------------------------


class TestPackage:
    def test_version_is_string(self):
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_public_api(self):
        assert BoxolFlower is not None
        assert GuiController is not None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestSacredCoords:
    def test_five_points(self):
        assert len(_SACRED_COORDS) == 5

    def test_contains_origin(self):
        assert (0, 0, 0) in _SACRED_COORDS

    def test_unit_cross_arms(self):
        """The four arms are at distance 1 from the origin in the XY plane."""
        arms = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
        for arm in arms:
            assert arm in _SACRED_COORDS


class TestBuildGrid:
    def test_sacred_count_always_five(self):
        """Sacred count is always 5 regardless of other params."""
        for layers in (1, 2, 3, 6):
            _, kinds = _build_grid(layers, 3.5, 3)
            assert int(np.sum(kinds == "sacred")) == 5

    def test_layers_1_petal_count(self):
        """layer=1 produces 6 unique petal coords."""
        _, kinds = _build_grid(1, 3.5, 3)
        assert int(np.sum(kinds == "petal")) == 6

    def test_layers_2_petal_count(self):
        _, kinds = _build_grid(2, 3.5, 3)
        assert int(np.sum(kinds == "petal")) == 18

    def test_layers_3_petal_count(self):
        _, kinds = _build_grid(3, 3.5, 3)
        assert int(np.sum(kinds == "petal")) == 36

    def test_default_petal_count(self):
        """Default geometry (layers=6): 126 unique petal voxels."""
        _, kinds = _build_grid(6, 3.5, 3)
        assert int(np.sum(kinds == "petal")) == 126

    def test_coords_shape(self):
        coords, kinds = _build_grid(6, 3.5, 3)
        assert coords.shape == (131, 3)
        assert kinds.shape == (131,)

    def test_sacred_coords_at_z_zero(self):
        """All 5 sacred voxels must sit at z=0."""
        coords, kinds = _build_grid(6, 3.5, 3)
        sacred_z = coords[kinds == "sacred", 2]
        np.testing.assert_array_equal(sacred_z, 0.0)


# ---------------------------------------------------------------------------
# BoxolFlower instantiation
# ---------------------------------------------------------------------------


class TestBoxolFlowerInit:
    def test_default_sacred_count(self):
        """Default params: exactly 5 sacred voxels."""
        assert make_flower().sacred_count == 5

    def test_default_petal_count(self):
        """Default params: exactly 126 petal voxels."""
        assert make_flower().petal_count == 126

    def test_default_total_count(self):
        """Default params: exactly 131 voxels total."""
        f = make_flower()
        assert f.voxel_count == 131
        assert f.voxel_count == f.sacred_count + f.petal_count

    def test_custom_layers_1(self):
        f = make_flower(layers=1)
        assert f.sacred_count == 5
        assert f.petal_count == 6
        assert f.voxel_count == 11

    def test_custom_layers_3(self):
        f = make_flower(layers=3)
        assert f.petal_count == 36
        assert f.voxel_count == 41

    def test_custom_params_stored(self):
        f = make_flower(layers=2, spacing=2.0, z_modulo=2)
        assert f.layers == 2
        assert f.spacing == pytest.approx(2.0)
        assert f.z_modulo == 2

    def test_seed_determinism(self):
        """Same seed → identical rem_dream noise."""
        f1 = BoxolFlower(seed=7)
        f2 = BoxolFlower(seed=7)
        r1 = f1.rem_dream()
        r2 = f2.rem_dream()
        # Both return the same status string
        assert r1 == r2
        np.testing.assert_array_equal(f1._offsets, f2._offsets)

    def test_different_seeds_give_different_noise(self):
        f1 = BoxolFlower(seed=1)
        f2 = BoxolFlower(seed=2)
        f1.rem_dream()
        f2.rem_dream()
        petal_mask = f1._kinds == "petal"
        assert not np.allclose(f1._offsets[petal_mask], f2._offsets[petal_mask])


# ---------------------------------------------------------------------------
# sensory_input
# ---------------------------------------------------------------------------


class TestSensoryInput:
    def test_stores_string(self):
        f = make_flower()
        f.sensory_input("hello world")
        assert f._input == "hello world"

    def test_stores_array(self):
        f = make_flower()
        arr = np.array([1.0, 2.0, 3.0])
        f.sensory_input(arr)
        np.testing.assert_array_equal(f._input, arr)

    def test_overwrite(self):
        f = make_flower()
        f.sensory_input("first")
        f.sensory_input("second")
        assert f._input == "second"

    def test_initially_none(self):
        f = make_flower()
        assert f._input is None


# ---------------------------------------------------------------------------
# pendulum_bloom
# ---------------------------------------------------------------------------


class TestPendulumBloom:
    def test_returns_dict_keys(self):
        f = make_flower()
        result = f.pendulum_bloom(ticks=3)
        assert {"offsets", "cot", "decisions"}.issubset(result.keys())

    def test_offset_shape(self):
        f = make_flower()
        result = f.pendulum_bloom(ticks=3)
        assert result["offsets"].shape == (f.voxel_count, 3)

    def test_cot_is_string(self):
        f = make_flower()
        result = f.pendulum_bloom(ticks=5)
        assert isinstance(result["cot"], str)
        assert len(result["cot"]) > 0

    def test_decisions_empty_without_input(self):
        """No executive decisions are generated when sensory input is absent."""
        f = make_flower()
        result = f.pendulum_bloom(ticks=5)
        assert result["decisions"] == []

    def test_decisions_populated_with_input(self):
        f = make_flower()
        f.sensory_input("neural collab")
        result = f.pendulum_bloom(ticks=3)
        assert len(result["decisions"]) == 3

    def test_phase_advances(self):
        f = make_flower()
        f.pendulum_bloom(ticks=1, step=0.5)
        assert f._phase == pytest.approx(0.5)
        f.pendulum_bloom(ticks=2, step=0.5)
        assert f._phase == pytest.approx(1.5)

    def test_sacred_voxels_not_displaced(self):
        """Sacred voxels must retain zero z-offset after bloom."""
        f = make_flower()
        f.pendulum_bloom(ticks=10)
        sacred_mask = f._kinds == "sacred"
        np.testing.assert_array_equal(f._offsets[sacred_mask, 2], 0.0)

    def test_returns_offset_copy(self):
        """Mutating the returned array must not affect internal state."""
        f = make_flower()
        result = f.pendulum_bloom(ticks=1)
        result["offsets"][:] = 999.0
        assert not np.any(f._offsets == 999.0)

    def test_cot_contains_pulse(self):
        f = make_flower()
        f.sensory_input("test")
        f.pendulum_bloom(ticks=2)
        result = f.pendulum_bloom(ticks=1)
        assert "pulse=" in result["cot"]


# ---------------------------------------------------------------------------
# rem_dream
# ---------------------------------------------------------------------------


class TestRemDream:
    def test_returns_string(self):
        f = make_flower()
        msg = f.rem_dream()
        assert isinstance(msg, str)
        assert "sacred" in msg.lower() or "bloodline" in msg.lower()

    def test_sacred_voxels_unperturbed(self):
        f = make_flower()
        before_sacred = f._offsets[f._kinds == "sacred"].copy()
        f.rem_dream()
        np.testing.assert_array_equal(
            f._offsets[f._kinds == "sacred"], before_sacred
        )

    def test_petals_perturbed_by_dream(self):
        f = BoxolFlower(seed=SEED)
        before = f._offsets.copy()
        f.rem_dream()
        petal_mask = f._kinds == "petal"
        assert not np.allclose(f._offsets[petal_mask], before[petal_mask])


# ---------------------------------------------------------------------------
# echo_ripple
# ---------------------------------------------------------------------------


class TestEchoRipple:
    def test_returns_array_of_correct_shape(self):
        f = make_flower()
        offsets = f.echo_ripple()
        assert offsets.shape == (f.voxel_count, 3)

    def test_zero_amplitude_no_change(self):
        f = make_flower()
        before = f._offsets.copy()
        f.echo_ripple(amplitude=0.0)
        np.testing.assert_array_almost_equal(f._offsets, before)

    def test_returns_copy(self):
        f = make_flower()
        offsets = f.echo_ripple(amplitude=1.0)
        offsets[:] = 0.0
        assert not np.all(f._offsets == 0.0)


# ---------------------------------------------------------------------------
# executive_decide
# ---------------------------------------------------------------------------


class TestExecutiveDecide:
    def test_returns_dict_with_expected_keys(self):
        f = make_flower()
        f.sensory_input("chess depth test")
        result = f.executive_decide()
        for key in (
            "depth", "input_preview", "priority", "risk",
            "urgency", "gravitational_pull", "pulse",
        ):
            assert key in result, f"Missing key: {key}"

    def test_priority_in_range(self):
        f = make_flower()
        f.sensory_input("test input")
        result = f.executive_decide()
        assert 0.0 <= result["priority"] <= 1.0

    def test_pulse_increments(self):
        f = make_flower()
        f.sensory_input("a")
        r1 = f.executive_decide()
        r2 = f.executive_decide()
        assert r2["pulse"] == r1["pulse"] + 1

    def test_no_input_returns_zero_scores(self):
        f = make_flower()
        result = f.executive_decide()
        assert result["priority"] == 0.0
        assert result["risk"] == 0.0

    def test_depth_stored_in_result(self):
        f = make_flower()
        f.sensory_input("test")
        result = f.executive_decide(depth=3)
        assert result["depth"] == 3

    def test_input_preview_truncated(self):
        f = make_flower()
        long_input = "x" * 100
        f.sensory_input(long_input)
        result = f.executive_decide()
        assert len(result["input_preview"]) <= 50

    def test_deterministic_with_seed(self):
        f1 = BoxolFlower(seed=99)
        f2 = BoxolFlower(seed=99)
        f1.sensory_input("same input")
        f2.sensory_input("same input")
        r1 = f1.executive_decide()
        r2 = f2.executive_decide()
        assert r1["priority"] == r2["priority"]
        assert r1["risk"] == r2["risk"]


# ---------------------------------------------------------------------------
# vassal_bus
# ---------------------------------------------------------------------------


class TestVassalBus:
    def test_returns_string(self):
        f = make_flower()
        msg = f.vassal_bus(1)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_stores_last_vassal(self):
        f = make_flower()
        f.vassal_bus(2)
        assert f._last_vassal == 2

    def test_all_four_buses(self):
        f = make_flower()
        for bus_id in (1, 2, 3, 4):
            msg = f.vassal_bus(bus_id)
            assert str(bus_id) in msg
            assert f._last_vassal == bus_id

    def test_unknown_bus_id(self):
        """Unknown bus IDs should not raise."""
        f = make_flower()
        msg = f.vassal_bus(99)
        assert "99" in msg


# ---------------------------------------------------------------------------
# self_heal
# ---------------------------------------------------------------------------


class TestSelfHeal:
    def test_returns_string(self):
        f = make_flower()
        msg = f.self_heal()
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_resets_offsets(self):
        f = make_flower()
        f.pendulum_bloom(ticks=10)
        f.self_heal()
        np.testing.assert_array_equal(f._offsets, 0.0)

    def test_resets_phase(self):
        f = make_flower()
        f.pendulum_bloom(ticks=10, step=0.5)
        f.self_heal()
        assert f._phase == 0.0

    def test_resets_pulse(self):
        f = make_flower()
        f.sensory_input("x")
        f.executive_decide()
        f.executive_decide()
        f.self_heal()
        assert f._pulse == 0

    def test_resets_input(self):
        f = make_flower()
        f.sensory_input("test")
        f.self_heal()
        assert f._input is None

    def test_resets_cot(self):
        f = make_flower()
        f.sensory_input("x")
        f.pendulum_bloom(ticks=1)
        assert f._last_cot != ""
        f.self_heal()
        assert f._last_cot == ""

    def test_resets_active(self):
        f = make_flower()
        f._active[:] = False
        f.self_heal()
        assert f._active.all()


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


class TestRender:
    def test_keys(self):
        data = make_flower().render()
        assert set(data.keys()) == {"coords", "kinds", "active"}

    def test_shapes(self):
        f = make_flower()
        data = f.render()
        n = f.voxel_count
        assert data["coords"].shape == (n, 3)
        assert data["kinds"].shape == (n,)
        assert data["active"].shape == (n,)

    def test_coords_include_offsets(self):
        f = make_flower()
        f.pendulum_bloom(ticks=3)
        data = f.render()
        expected = f._coords + f._offsets
        np.testing.assert_array_almost_equal(data["coords"], expected)

    def test_returns_copies(self):
        f = make_flower()
        data = f.render()
        data["coords"][:] = 0.0
        data["kinds"][:] = "x"
        data["active"][:] = False
        # Internal state must not change.
        assert not np.all(f._coords == 0.0)
        assert "sacred" in f._kinds
        assert f._active.all()

    def test_sacred_kinds_present(self):
        data = make_flower().render()
        assert "sacred" in data["kinds"]
        assert "petal" in data["kinds"]

    def test_active_all_true_at_init(self):
        data = make_flower().render()
        assert data["active"].all()


# ---------------------------------------------------------------------------
# GuiController — headless environment check
# ---------------------------------------------------------------------------


class TestGuiController:
    def test_raises_when_gui_unavailable(self):
        """In a headless CI environment GuiController must raise RuntimeError."""
        from boxol_flower.boxol import _GUI_AVAILABLE

        if _GUI_AVAILABLE:
            pytest.skip("GUI is available in this environment")

        f = make_flower()
        with pytest.raises(RuntimeError, match="GUI unavailable"):
            GuiController(f)
