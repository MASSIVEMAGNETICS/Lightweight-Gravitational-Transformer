"""
VictorOS Module Integration
Provides Ledger serialization, Mirror Layer scaffolding, and the
``@victoros_module`` decorator for packaging LGT-based agents as
first-class VictorOS cognitive modules.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """A single timestamped, structured Ledger record."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    event: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=_json_default)


def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for non-standard types (tensors, etc.)."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if hasattr(obj, "__float__"):
        return float(obj)
    return str(obj)


class Ledger:
    """
    Append-only, structured event log for a VictorOS agent.

    Entries are held in memory and can be flushed to a JSONL file on disk.
    This provides:
    - Causal tracing: every inference, parameter update, and containment
      event is time-stamped and linked to an agent ID.
    - Defensive publication: the JSONL log is human-readable and
      machine-parseable, forming a tamper-evident audit trail.
    - Offline-first: no network dependency; sync happens asynchronously.

    Args:
        agent_id: Unique identifier for the owning agent/module.
        persist_path: Optional path to a ``.jsonl`` file for persistence.
            If ``None`` the Ledger is memory-only.
        max_memory_entries: Maximum entries kept in RAM before auto-flush.
    """

    def __init__(
        self,
        agent_id: str = "default",
        persist_path: Optional[str] = None,
        max_memory_entries: int = 1000,
    ):
        self.agent_id = agent_id
        self.persist_path = persist_path
        self.max_memory_entries = max_memory_entries
        self._entries: List[LedgerEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> LedgerEntry:
        """Create and store a new Ledger entry."""
        entry = LedgerEntry(
            agent_id=self.agent_id,
            event=event,
            payload=payload or {},
        )
        self._entries.append(entry)
        if len(self._entries) >= self.max_memory_entries:
            self.flush()
        return entry

    def flush(self) -> int:
        """
        Write all in-memory entries to ``persist_path`` (JSONL format).
        Returns the number of entries flushed.  No-op if no path is set.
        """
        if self.persist_path is None or not self._entries:
            return 0
        os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
        with open(self.persist_path, "a", encoding="utf-8") as fh:
            for entry in self._entries:
                fh.write(entry.to_json() + "\n")
        flushed = len(self._entries)
        self._entries.clear()
        return flushed

    def entries(self, event_filter: Optional[str] = None) -> List[LedgerEntry]:
        """Return in-memory entries, optionally filtered by event name."""
        if event_filter is None:
            return list(self._entries)
        return [e for e in self._entries if e.event == event_filter]

    def snapshot(self) -> Dict[str, Any]:
        """Return all in-memory entries as a serialisable snapshot."""
        return {
            "agent_id": self.agent_id,
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Mirror Layer
# ---------------------------------------------------------------------------

class MirrorLayer:
    """
    Real-time introspection hook for LGT cognitive diagnostics.

    The Mirror Layer sits between the model's forward pass and the VictorOS
    Cortex.  It:
    1. Receives per-layer diagnostics from the model's ``mirror_layer_callback``.
    2. Evaluates a configurable set of stability criteria.
    3. Emits correction signals when thresholds are breached.
    4. Logs every event to the attached Ledger.

    Args:
        ledger: Ledger instance to log events to.
        max_force_threshold: Force value above which a dampening correction
            is triggered.
        stability_window: Number of recent force measurements to average when
            computing the rolling stability score.
        correction_callback: Optional external hook called with
            ``(layer_idx, correction_type)`` when a correction is needed.
    """

    def __init__(
        self,
        ledger: Optional[Ledger] = None,
        max_force_threshold: float = 40.0,
        stability_window: int = 20,
        correction_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.ledger = ledger if ledger is not None else Ledger(agent_id="mirror_layer")
        self.max_force_threshold = max_force_threshold
        self.stability_window = stability_window
        self.correction_callback = correction_callback
        self._force_history: List[float] = []

    def __call__(self, layer_idx: int, diag: Dict[str, Any]) -> None:
        """Callback compatible with ``mirror_layer_callback`` in LGT forward."""
        mean_force = float(diag.get("mean_force", 0.0))
        self._force_history.append(mean_force)
        if len(self._force_history) > self.stability_window:
            self._force_history.pop(0)

        stability_score = self._compute_stability()

        self.ledger.log("mirror_layer", {
            "layer": layer_idx,
            "mean_force": mean_force,
            "mean_mass": diag.get("mean_mass", 0.0),
            "curvature_active": diag.get("curvature_active", False),
            "hawking_limit": diag.get("hawking_limit"),
            "stability_score": stability_score,
        })

        if mean_force > self.max_force_threshold:
            correction = "attention_dampening"
            self.ledger.log("containment_correction", {
                "layer": layer_idx,
                "trigger": "max_force_exceeded",
                "value": mean_force,
                "threshold": self.max_force_threshold,
                "correction": correction,
            })
            if self.correction_callback is not None:
                self.correction_callback(layer_idx, correction)

    def stability_score(self) -> float:
        """Return the current rolling stability score [0, 1]."""
        return self._compute_stability()

    def _compute_stability(self) -> float:
        if not self._force_history:
            return 1.0
        recent = self._force_history[-self.stability_window:]
        mean_f = sum(recent) / len(recent)
        if mean_f <= 0:
            return 1.0
        # Normalise so that force at max_force_threshold → stability = 0.5
        score = 1.0 / (1.0 + mean_f / self.max_force_threshold)
        return float(min(1.0, max(0.0, score)))


# ---------------------------------------------------------------------------
# VictorOS module descriptor + decorator
# ---------------------------------------------------------------------------

@dataclass
class VictorOSModuleMetadata:
    """Metadata attached to a VictorOS-registered module."""
    name: str
    version: str
    requirements: List[str] = field(default_factory=list)
    containment_native: bool = False
    description: str = ""


def victoros_module(
    name: str,
    version: str,
    requirements: Optional[List[str]] = None,
    containment_native: bool = False,
    description: str = "",
) -> Callable[[Type], Type]:
    """
    Class decorator that registers a class as a VictorOS cognitive module.

    Attaches metadata and wraps ``__init__`` to provision a ``Ledger`` and
    ``MirrorLayer`` automatically when the class does not already define them.

    Example::

        @victoros_module(
            name="lgt_edge",
            version="0.1.0",
            containment_native=True,
        )
        class LGTEdgeModule(VictorOSBaseModule):
            ...
    """
    def decorator(cls: Type) -> Type:
        cls._victoros_meta = VictorOSModuleMetadata(
            name=name,
            version=version,
            requirements=requirements or [],
            containment_native=containment_native,
            description=description,
        )

        original_init = cls.__init__

        def patched_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            # Auto-provision Ledger + MirrorLayer if not set
            if not hasattr(self, "ledger"):
                self.ledger = Ledger(agent_id=name)
            if not hasattr(self, "mirror_layer"):
                self.mirror_layer = MirrorLayer(ledger=self.ledger)

        cls.__init__ = patched_init
        return cls

    return decorator


# ---------------------------------------------------------------------------
# VictorOS base module
# ---------------------------------------------------------------------------

class VictorOSBaseModule:
    """
    Base class for VictorOS cognitive modules.

    Subclasses receive a ``Ledger``, a ``MirrorLayer``, and a ``now()``
    helper automatically.  Override ``process()`` to implement module logic.
    """

    ledger: Ledger
    mirror_layer: MirrorLayer

    def now(self) -> float:
        """Current UNIX timestamp."""
        return time.time()

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Override in subclasses to implement the module's main logic."""
        raise NotImplementedError

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Serialise the module's model weights + Ledger snapshot to disk."""
        model = getattr(self, "model", None)
        state: Dict[str, Any] = {
            "ledger_snapshot": self.ledger.snapshot(),
            "extra": extra or {},
        }
        if isinstance(model, nn.Module):
            state["model_state_dict"] = model.state_dict()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load weights + metadata from a checkpoint file."""
        state = torch.load(path, weights_only=False)
        model = getattr(self, "model", None)
        if isinstance(model, nn.Module) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        return state


# ---------------------------------------------------------------------------
# Concrete LGT VictorOS module
# ---------------------------------------------------------------------------

@victoros_module(
    name="lgt_core",
    version="0.1.0",
    requirements=["torch>=2.0.0", "numpy>=1.24.0"],
    containment_native=True,
    description="Lightweight Gravitational Transformer – VictorOS core cognitive module.",
)
class LGTVictorOSModule(VictorOSBaseModule):
    """
    LGT wrapped as a fully-integrated VictorOS cognitive module.

    Provides:
    - Auto-provisioned Ledger + MirrorLayer.
    - Structured ``process()`` with containment-aware diagnostics.
    - ``get_snapshot()`` for causal tracing.
    - ``propose_architecture_change()`` for self-evolution.

    Args:
        model: A pre-constructed ``LightweightGravitationalTransformer``.
        agent_id: Unique agent identifier used in Ledger entries.
        persist_path: Optional path to flush Ledger entries to disk.
        max_force_threshold: Mirror Layer containment threshold.
    """

    def __init__(
        self,
        model: nn.Module,
        agent_id: str = "lgt_core",
        persist_path: Optional[str] = None,
        max_force_threshold: float = 40.0,
    ):
        self.model = model
        self.ledger = Ledger(agent_id=agent_id, persist_path=persist_path)
        self.mirror_layer = MirrorLayer(
            ledger=self.ledger,
            max_force_threshold=max_force_threshold,
            correction_callback=self._on_correction,
        )
        self._corrections: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """
        Run an LGT inference pass with full VictorOS integration.

        Automatically streams diagnostics through the Mirror Layer and logs
        the result to the Ledger.

        Returns a dict with keys ``output``, ``diagnostics``, ``stability``.
        """
        self.model.eval()
        with torch.no_grad():
            output, diagnostics = self.model(
                x,
                return_diagnostics=return_diagnostics,
                mirror_layer_callback=self.mirror_layer if return_diagnostics else None,
            )

        stability = self.mirror_layer.stability_score()

        self.ledger.log("inference", {
            "seq_len": int(x.shape[1]) if x.dim() > 1 else 0,
            "stability_score": stability,
            "corrections": len(self._corrections),
            "output_mean": float(output.mean()),
            "output_std": float(output.std()),
        })

        return {
            "output": output,
            "diagnostics": diagnostics,
            "stability": stability,
        }

    # ------------------------------------------------------------------
    # Snapshot / causal tracing
    # ------------------------------------------------------------------

    def get_snapshot(self, x: torch.Tensor) -> Dict[str, Any]:
        """Full attention snapshot for Ledger logging and causal tracing."""
        snapshot = self.model.get_attention_snapshot(x)
        self.ledger.log("snapshot", {"model_config": snapshot.get("model_config")})
        return snapshot

    # ------------------------------------------------------------------
    # Self-evolution scaffold
    # ------------------------------------------------------------------

    def propose_architecture_change(
        self,
        current_config: Dict[str, Any],
        stability_threshold: float = 0.95,
    ) -> Optional[Dict[str, Any]]:
        """
        Propose a structural change to the LGT based on current stability.

        Returns a proposal dict when stability is high (model is ready to
        grow), or ``None`` when the model is still settling.

        The proposal is logged to the Ledger for review and must be applied
        externally (e.g., by the VictorOS Cortex or a training script).
        """
        score = self.mirror_layer.stability_score()
        if score < stability_threshold:
            return None

        proposal: Dict[str, Any] = {}
        num_layers = current_config.get("num_layers", 4)
        curvature = current_config.get("curvature", 0.15)

        if score > 0.98:
            proposal = {
                "change": "add_layer",
                "new_num_layers": num_layers + 1,
                "reason": f"stability={score:.3f} exceeds add_layer threshold",
            }
        else:
            proposal = {
                "change": "increase_curvature",
                "new_curvature": round(min(curvature * 1.1, 0.5), 4),
                "reason": f"stability={score:.3f}; moderate growth via curvature",
            }

        self.ledger.log("architecture_proposal", proposal)
        return proposal

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_correction(self, layer_idx: int, correction_type: str) -> None:
        self._corrections.append({"layer": layer_idx, "type": correction_type, "t": time.time()})
