"""
Core implementation of the Boxol Flower simulation.

Refactored from ``BoxolFlowerMonolith v1.6`` (boxol_flower_mono.py) into a
clean, importable, headless-first Python class with optional GUI support.

Classes
-------
BoxolFlower
    Pure-NumPy headless computation core.  Generates a fractal voxel bloom
    with a fixed 5-point sacred cross at the centre (gold) and configurable
    hexagonal petal rings (magenta).  Exposes all simulation methods without
    requiring a display, tkinter, or matplotlib.

GuiController
    Optional Tkinter / Matplotlib 3-D scatter visualiser that wraps a
    ``BoxolFlower`` instance.  Raises :exc:`RuntimeError` when the required
    GUI libraries are unavailable or no display is reachable.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__version__ = "1.6.0"

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional GUI dependencies (tkinter + matplotlib)
# ---------------------------------------------------------------------------
_GUI_AVAILABLE: bool = False
_tkinter: Any = None
_ttk: Any = None
_Figure: Any = None
_FigureCanvasTkAgg: Any = None

try:
    import tkinter as _tkinter  # type: ignore[no-redef]
    from tkinter import ttk as _ttk  # type: ignore[no-redef]
    import matplotlib as _mpl  # type: ignore

    _mpl.use("TkAgg")
    from matplotlib.figure import Figure as _Figure  # type: ignore[no-redef]
    from matplotlib.backends.backend_tkagg import (  # type: ignore
        FigureCanvasTkAgg as _FigureCanvasTkAgg,
    )
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401

    _GUI_AVAILABLE = True
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Optional torch dependency
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE: bool = False
try:
    import torch as _torch  # type: ignore  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
#: The fixed 5-point cross that forms the sacred centre of the flower.
#: These coordinates are taken directly from BoxolFlowerMonolith v1.6.
_SACRED_COORDS: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (-1, 0, 0),
    (0, -1, 0),
]

# ---------------------------------------------------------------------------
# Configurable defaults  (matching BoxolFlowerMonolith v1.6 geometry)
# ---------------------------------------------------------------------------
DEFAULT_LAYERS: int = 6      # petal rings: range(1, layers+1)
DEFAULT_SPACING: float = 3.5  # radial multiplier: layer * spacing
DEFAULT_Z_MODULO: int = 3    # z = layer % z_modulo

# ---------------------------------------------------------------------------
# Vassal bus registry
# ---------------------------------------------------------------------------
_VASSAL_NAMES: Dict[int, str] = {
    1: "Files",
    2: "World",
    3: "Computers",
    4: "APIs",
}

# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------


def _build_grid(
    layers: int,
    spacing: float,
    z_modulo: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the voxel grid matching the ``BoxolFlowerMonolith v1.6`` geometry.

    Sacred voxels occupy the 5-point cross at ``z=0``; petal voxels fill
    concentric rings around that centre, with ``z = layer % z_modulo``.
    Coordinate collisions are resolved by a last-write-wins dict (matching
    the original monolith behaviour), except that sacred coords always
    retain their classification.

    Parameters
    ----------
    layers:
        Number of petal rings, equivalent to ``range(1, layers+1)`` in the
        original (default 6).
    spacing:
        Radial multiplier: ring ``r`` is placed at radius ``r * spacing``
        (default 3.5, matching the original hardcoded ``3.5``).
    z_modulo:
        Integer divisor for the z coordinate: ``z = layer % z_modulo``
        (default 3).

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
        Voxel positions as floats (values are integer-valued due to
        ``int()`` truncation, preserving the original visual layout).
    kinds : np.ndarray of dtype object, shape (N,)
        ``'sacred'`` for the 5-point cross, ``'petal'`` for everything else.
    """
    sacred_set = set(_SACRED_COORDS)
    grid: Dict[Tuple[int, int, int], str] = {}

    # Place sacred centre first so petals cannot overwrite it.
    for coord in _SACRED_COORDS:
        grid[coord] = "sacred"

    # Petal rings (mirrors BoxolFlowerMonolith.build_sacred_flower)
    for layer in range(1, layers + 1):
        z = layer % z_modulo
        for i in range(6 * layer):
            angle = i * 2 * math.pi / (6 * layer)
            x = int(layer * spacing * math.cos(angle))
            y = int(layer * spacing * math.sin(angle))
            coord = (x, y, z)
            if coord not in sacred_set:
                grid[coord] = "petal"

    coords_list = list(grid.keys())
    kinds_list = list(grid.values())

    coords = np.array(coords_list, dtype=float)
    kinds = np.array(kinds_list, dtype=object)
    return coords, kinds


# ---------------------------------------------------------------------------
# BoxolFlower
# ---------------------------------------------------------------------------


class BoxolFlower:
    """
    Fractal 3-D voxel-bloom simulation — headless, importable core.

    Mirrors the behaviour of ``BoxolFlowerMonolith v1.6`` without requiring
    a display, tkinter, or matplotlib.  All geometry is computed with NumPy.

    Voxel counts for default parameters (``layers=6``, ``spacing=3.5``,
    ``z_modulo=3``)::

        sacred  =   5   (fixed 5-point cross at z=0)
        petal   = 126   (hexagonal rings 1-6)
        total   = 131

    Parameters
    ----------
    layers:
        Number of hexagonal petal rings (default 6).
    spacing:
        Radial multiplier for petal ring placement (default 3.5).
    z_modulo:
        Modulo divisor for the z coordinate of each ring (default 3).
    seed:
        Optional integer seed for reproducible RNG behaviour.
    """

    def __init__(
        self,
        layers: int = DEFAULT_LAYERS,
        spacing: float = DEFAULT_SPACING,
        z_modulo: int = DEFAULT_Z_MODULO,
        seed: Optional[int] = None,
    ) -> None:
        self.layers = layers
        self.spacing = spacing
        self.z_modulo = z_modulo
        self._rng = np.random.default_rng(seed)

        self._coords, self._kinds = _build_grid(layers, spacing, z_modulo)
        self._offsets: np.ndarray = np.zeros_like(self._coords)
        self._phase: float = 0.0
        self._pulse: int = 0
        self._input: Optional[Any] = None
        self._last_cot: str = ""
        self._last_decision: Dict[str, Any] = {}
        self._last_vassal: Optional[int] = None
        self._active: np.ndarray = np.ones(len(self._coords), dtype=bool)

        _log.info(
            "BoxolFlower ready: layers=%d spacing=%.2f z_modulo=%d "
            "voxels=%d (sacred=%d petal=%d)",
            layers,
            spacing,
            z_modulo,
            self.voxel_count,
            self.sacred_count,
            self.petal_count,
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def sacred_count(self) -> int:
        """Number of sacred (central cross) voxels."""
        return int(np.sum(self._kinds == "sacred"))

    @property
    def petal_count(self) -> int:
        """Number of petal voxels."""
        return int(np.sum(self._kinds == "petal"))

    @property
    def voxel_count(self) -> int:
        """Total voxel count (sacred + petal)."""
        return len(self._coords)

    # ------------------------------------------------------------------
    # Public simulation methods
    # ------------------------------------------------------------------

    def sensory_input(self, data: Any) -> None:
        """
        Accept an external stimulus and store it for processing.

        Mirrors the ``live_input_update`` method of the original monolith.

        Parameters
        ----------
        data:
            Arbitrary stimulus (text string, scalar, NumPy array, etc.).
        """
        self._input = data
        _log.debug("sensory input received: %s", repr(data)[:80])

    def pendulum_bloom(
        self, ticks: int = 5, step: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run *ticks* pendulum phases with Chain-of-Thought logging.

        Each tick advances the internal phase, applies a sinusoidal
        z-displacement to petal voxels, and records an executive decision
        snapshot (if sensory input is present).  This mirrors the
        ``pendulum_cot`` / 5-tick behaviour of the original monolith.

        Parameters
        ----------
        ticks:
            Number of phase ticks to execute (default 5).
        step:
            Phase increment per tick in radians (default 0.1).

        Returns
        -------
        dict with keys:

        ``'offsets'``
            Position offsets after the final tick, shape ``(N, 3)``.
        ``'cot'``
            Chain-of-Thought log string.
        ``'decisions'``
            List of decision dicts, one per tick (empty when no input set).
        """
        decisions: List[Dict[str, Any]] = []
        petal_mask = self._kinds == "petal"
        dist = np.sqrt(
            self._coords[:, 0] ** 2 + self._coords[:, 1] ** 2
        )

        for _ in range(ticks):
            self._phase += step
            self._offsets[:, 2] = np.where(
                petal_mask,
                np.sin(self._phase + dist * 0.1) * self.spacing * 0.25,
                0.0,
            )
            if self._input is not None:
                decisions.append(self.executive_decide())

        cot = (
            f"WHO: BoxolFlower | "
            f"WHAT: {ticks}-tick pendulum bloom | "
            f"WHY: Sensory integration + sacred centre dominant | "
            f"WHEN: pulse={self._pulse} | "
            f"WHERE: Flower centre + vassal buses"
        )
        self._last_cot = cot
        _log.debug("pendulum_bloom ticks=%d phase=%.3f", ticks, self._phase)
        return {
            "offsets": self._offsets.copy(),
            "cot": cot,
            "decisions": decisions,
        }

    def rem_dream(self) -> str:
        """
        Compress memory and anchor identity to the sacred bloodline centre.

        Adds small Gaussian noise to petal offsets (dream perturbation) then
        returns a status string.  Sacred voxels are never perturbed.

        Returns
        -------
        str
            Human-readable status message.
        """
        petal_mask = (self._kinds == "petal")[:, np.newaxis]
        noise = self._rng.normal(
            0.0, self.spacing * 0.02, self._offsets.shape
        )
        self._offsets += np.where(petal_mask, noise, 0.0)
        msg = (
            "REM DREAM: Memory compressed, identity anchored in sacred "
            "bloodline centre, streams consolidated"
        )
        _log.debug("rem_dream completed")
        return msg

    def echo_ripple(
        self, amplitude: float = 1.0, speed: float = 1.0
    ) -> np.ndarray:
        """
        Propagate a radial wave outward from the sacred centre.

        The z-offset of each voxel is shifted by a sinusoidal function of
        its distance from the origin, modulated by the current phase.

        Parameters
        ----------
        amplitude:
            Peak displacement (default 1.0).
        speed:
            Phase-shift scale factor per unit distance (default 1.0).

        Returns
        -------
        np.ndarray
            Copy of the current position offsets, shape ``(N, 3)``.
        """
        dist = np.sqrt(np.sum(self._coords ** 2, axis=1))
        ripple = amplitude * np.sin(self._phase - speed * dist)
        self._offsets[:, 2] += ripple
        _log.debug(
            "echo_ripple amplitude=%.2f speed=%.2f", amplitude, speed
        )
        return self._offsets.copy()

    def executive_decide(
        self, depth: int = 5
    ) -> Dict[str, Any]:
        """
        Run a chess-depth gravitational executive decision.

        Mirrors the ``process_chess_decide`` method of the original monolith.
        Scores are generated from the seeded RNG so results are reproducible
        when a ``seed`` was passed to the constructor.

        Parameters
        ----------
        depth:
            Lookahead depth for the chess-style evaluation (default 5).

        Returns
        -------
        dict with keys: ``depth``, ``input_preview``, ``priority``,
        ``risk``, ``urgency``, ``gravitational_pull``, ``pulse``.

        Notes
        -----
        Returns a neutral zero-score result when no sensory input has been
        provided (matching the original ``if not self.current_input: return``
        guard).
        """
        text = str(self._input) if self._input is not None else ""
        if not text:
            _log.debug("executive_decide skipped: no sensory input")
            return {
                "depth": depth,
                "input_preview": "",
                "priority": 0.0,
                "risk": 0.0,
                "urgency": 0.0,
                "gravitational_pull": 0.0,
                "pulse": self._pulse,
            }

        base = float(self._rng.uniform(0.7, 1.0)) * (len(text) / 50.0)
        grav = float(self._rng.uniform(0.3, 0.8))
        score = min(base + grav, 1.0)

        result: Dict[str, Any] = {
            "depth": depth,
            "input_preview": text[:50],
            "priority": round(score, 4),
            "risk": round(float(self._rng.uniform(0.05, 0.2)), 4),
            "urgency": round(float(self._rng.uniform(0.7, 1.0)), 4),
            "gravitational_pull": round(
                float(self._rng.uniform(0.7, 0.95)), 4
            ),
            "pulse": self._pulse,
        }
        self._last_decision = result
        self._pulse += 1
        _log.debug(
            "executive_decide depth=%d priority=%.4f pulse=%d",
            depth,
            result["priority"],
            result["pulse"],
        )
        return result

    def vassal_bus(self, bus_id: int) -> str:
        """
        Activate a vassal bus for external interaction.

        Mirrors the ``vassal_bus`` method of the original monolith.
        Logs the activation at INFO level; no blocking print is emitted.

        Parameters
        ----------
        bus_id:
            Bus identifier (1=Files, 2=World, 3=Computers, 4=APIs).

        Returns
        -------
        str
            Status message describing the activated bus.
        """
        name = _VASSAL_NAMES.get(bus_id, f"Bus-{bus_id}")
        msg = (
            f"VASSAL BUS {bus_id} ({name}) ACTIVATED — external action "
            f"routed safely under bloodline executive control"
        )
        self._last_vassal = bus_id
        _log.info("vassal_bus %d (%s) activated", bus_id, name)
        return msg

    def self_heal(self) -> str:
        """
        Prune high-risk petals and reset state to baseline.

        Mirrors the ``self_heal`` method of the original monolith.
        Resets all position offsets, animation phase, activity mask, and
        sensory input.

        Returns
        -------
        str
            Status message.
        """
        self._offsets[:] = 0.0
        self._phase = 0.0
        self._pulse = 0
        self._active[:] = True
        self._input = None
        self._last_cot = ""
        self._last_decision = {}
        self._last_vassal = None
        msg = (
            "Self-healing: high-risk petals pruned, bloodline + executive "
            "centre reinforced, vassal buses re-aligned"
        )
        _log.info("self_heal: state reset to baseline")
        return msg

    def render(self) -> Dict[str, np.ndarray]:
        """
        Return a snapshot of the current simulation state as plain arrays.

        Pure-data method — does **not** open a window.
        For interactive display, use :class:`GuiController`.

        Returns
        -------
        dict with keys:

        ``'coords'``
            Effective voxel positions (base + offsets), shape ``(N, 3)``.
        ``'kinds'``
            Voxel type array (``'sacred'`` | ``'petal'``), shape ``(N,)``.
        ``'active'``
            Boolean activity mask, shape ``(N,)``.
        """
        return {
            "coords": self._coords + self._offsets,
            "kinds": self._kinds.copy(),
            "active": self._active.copy(),
        }


# ---------------------------------------------------------------------------
# GuiController  (requires tkinter + matplotlib + live display)
# ---------------------------------------------------------------------------


class GuiController:
    """
    Tkinter / Matplotlib 3-D scatter visualiser for :class:`BoxolFlower`.

    Replicates the GUI of ``BoxolFlowerMonolith v1.6`` using the refactored
    :class:`BoxolFlower` backend.  Only functional when ``tkinter`` and
    ``matplotlib`` are installed *and* a live display is available.

    Sacred voxels are rendered in **gold**; petal voxels in **magenta**.

    Parameters
    ----------
    flower:
        The :class:`BoxolFlower` instance to visualise.
    """

    _SACRED_COLOR = "gold"
    _PETAL_COLOR = "magenta"
    _WINDOW_SIZE = "1300x950"
    _TITLE = (
        "BOXOL FLOWER v1.6 — Gravitational Chess Executive + "
        "4 Vassal Buses + 5-Tick CoT Logging"
    )

    def __init__(self, flower: BoxolFlower) -> None:  # pragma: no cover
        if not _GUI_AVAILABLE:
            raise RuntimeError(
                "GUI unavailable: tkinter or matplotlib not installed, "
                "or no display detected.  Use headless mode instead."
            )
        self._flower = flower
        self._root = _tkinter.Tk()
        self._root.title(self._TITLE)
        self._root.geometry(self._WINDOW_SIZE)

        self._fig = _Figure(figsize=(14, 9))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._canvas = _FigureCanvasTkAgg(self._fig, master=self._root)
        self._canvas.get_tk_widget().pack(fill=_tkinter.BOTH, expand=True)

        self._build_controls()
        self.update()
        _log.info("GuiController created")

    def _build_controls(self) -> None:  # pragma: no cover
        """Construct the input frame, buttons, and status labels."""
        # Sensory input row
        inp_frame = _ttk.Frame(self._root)
        inp_frame.pack(fill=_tkinter.X, pady=8)
        _ttk.Label(
            inp_frame,
            text="LIVE SENSORY INPUT -> TYPE HERE:",
            font=("Courier", 11, "bold"),
        ).pack(side=_tkinter.LEFT, padx=10)
        self._input_entry = _ttk.Entry(
            inp_frame, width=90, font=("Courier", 10)
        )
        self._input_entry.pack(
            side=_tkinter.LEFT, fill=_tkinter.X, expand=True, padx=5
        )
        self._input_entry.bind("<KeyRelease>", self._on_key)

        # Button row
        btn_frame = _ttk.Frame(self._root)
        btn_frame.pack(fill=_tkinter.X, pady=5)
        buttons = [
            ("PROCESS + EXECUTIVE CHESS DECIDE", self._do_decide),
            ("5-TICK PENDULUM CoT + HOLO LOG", self._do_pendulum),
            ("REM DREAM + ANCHOR", self._do_dream),
            ("VASSAL BUS 1 (Files)", lambda: self._do_vassal(1)),
            ("VASSAL BUS 2 (World)", lambda: self._do_vassal(2)),
            ("VASSAL BUS 3 (Computers)", lambda: self._do_vassal(3)),
            ("VASSAL BUS 4 (APIs)", lambda: self._do_vassal(4)),
        ]
        for label, cmd in buttons:
            _ttk.Button(btn_frame, text=label, command=cmd).pack(
                side=_tkinter.LEFT, padx=5
            )

        # Status and decision labels
        self._status_lbl = _tkinter.Label(
            self._root,
            text=(
                "BOXOL FLOWER v1.6 LIVE — Gravitational Chess Executive + "
                "4 Vassal Buses + 5-Tick CoT Holo Logging"
            ),
            fg="#ff00ff",
            font=("Courier", 12, "bold"),
        )
        self._status_lbl.pack(pady=5)

        self._decision_lbl = _tkinter.Label(
            self._root,
            text="EXECUTIVE OUTPUT + CoT LOG: Awaiting input...",
            fg="cyan",
            font=("Courier", 11),
            wraplength=1100,
            justify="left",
        )
        self._decision_lbl.pack(pady=5)

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

    def _on_key(self, _event: Any = None) -> None:  # pragma: no cover
        self._flower.sensory_input(self._input_entry.get().strip())

    def _do_decide(self) -> None:  # pragma: no cover
        result = self._flower.executive_decide()
        if not result["input_preview"]:
            return
        text = (
            f"CHESS-DEPTH EXECUTIVE (depth {result['depth']}) | "
            f"Input: '{result['input_preview']}...'\n"
            f"-> Priority {result['priority']} | "
            f"Risk {result['risk']} | "
            f"Urgency {result['urgency']}\n"
            f"-> Gravitational Pull: {result['gravitational_pull']} "
            f"(centre seed dominant) | pulse={result['pulse']}"
        )
        self._decision_lbl.config(text=text, fg="#00ff00")
        self._status_lbl.config(
            text=(
                f"EXECUTIVE DECIDED — gravitational chess evaluation "
                f"complete (pulse {result['pulse']})"
            )
        )
        self.update()

    def _do_pendulum(self) -> None:  # pragma: no cover
        result = self._flower.pendulum_bloom(ticks=5)
        cot = result["cot"]
        self._decision_lbl.config(
            text=f"5-TICK CoT HOLO LOG: {cot[:120]}...", fg="yellow"
        )
        self._status_lbl.config(
            text=(
                "5-Tick pendulum CoT + holographic neuromorphic log complete"
            )
        )
        self.update()

    def _do_dream(self) -> None:  # pragma: no cover
        msg = self._flower.rem_dream()
        self._decision_lbl.config(text=msg)
        self._status_lbl.config(
            text="Dream cycle finished — 7D holographic anchor locked"
        )
        self.update()

    def _do_vassal(self, bus_id: int) -> None:  # pragma: no cover
        msg = self._flower.vassal_bus(bus_id)
        self._decision_lbl.config(text=msg)
        self.update()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update(self) -> None:  # pragma: no cover
        """Re-draw the 3-D scatter plot with the current flower state."""
        data = self._flower.render()
        coords = data["coords"]
        kinds = data["kinds"]

        self._ax.clear()
        sacred_m = kinds == "sacred"
        petal_m = kinds == "petal"

        self._ax.scatter(
            coords[sacred_m, 0],
            coords[sacred_m, 1],
            coords[sacred_m, 2],
            c=self._SACRED_COLOR,
            s=110,
            label="sacred",
            depthshade=True,
        )
        self._ax.scatter(
            coords[petal_m, 0],
            coords[petal_m, 1],
            coords[petal_m, 2],
            c=self._PETAL_COLOR,
            s=60,
            label="petal",
            depthshade=True,
        )
        self._ax.set_title(
            "BOXOL FLOWER v1.6 — COMPLETE NEURAL NET | "
            "Gravitational Chess + 4 Vassals + CoT Holo Active"
        )
        self._ax.legend()
        self._canvas.draw()

    def run(self) -> None:  # pragma: no cover
        """Start the Tkinter event loop."""
        _log.info("GuiController event loop starting")
        self._root.mainloop()
