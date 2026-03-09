"""
Training Loop with Containment Protocol
Provides a physics-aware training loop for the LGT with built-in safety
constraints derived from the Black Hole Framework:

- ContainmentProtocol: gradient-norm bounds, Bekenstein entropy cap,
  attention-force ceiling, stability scoring, and architecture-mutation
  proposals.
- MetaCurvatureScheduler: adjusts each layer's curvature parameter using
  a light meta-gradient signal derived from validation loss.
- TrainingLoop: orchestrates optimisation with the above constraints and
  optional Mirror Layer / Ledger integration.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ---------------------------------------------------------------------------
# Containment Protocol
# ---------------------------------------------------------------------------

@dataclass
class ContainmentConfig:
    """
    Configuration for the ContainmentProtocol safety checks.

    Attributes:
        max_grad_norm: Maximum L2-norm of gradients; triggers clipping.
        max_attention_force: Maximum mean gravitational force across layers;
            triggers dampening if exceeded during training.
        bekenstein_lambda: Regularisation weight for the Bekenstein entropy
            penalty (encourages compressed, low-entropy representations).
        min_loss: Hard lower bound on training loss; triggers early stopping
            if loss falls below this (potential mode collapse / overfitting).
        max_loss: Hard upper bound; triggers early stopping on divergence.
        stability_ema_alpha: Exponential moving average coefficient for the
            rolling stability score.
        enable_architecture_proposals: When True, the protocol emits
            architecture-change proposals when stability is very high.
        stability_proposal_threshold: Stability score [0, 1] above which a
            structural proposal is generated.
        proposal_min_interval: Minimum number of training steps between
            successive architecture-mutation proposals.
    """
    max_grad_norm: float = 1.0
    max_attention_force: float = 40.0
    bekenstein_lambda: float = 1e-4
    min_loss: float = 1e-8
    max_loss: float = 1e4
    stability_ema_alpha: float = 0.05
    enable_architecture_proposals: bool = True
    stability_proposal_threshold: float = 0.95
    proposal_min_interval: int = 100


class ContainmentProtocol:
    """
    Real-time safety and stability guard for LGT training.

    The protocol wraps each training step and:
    1. **Gradient clipping** – hard cap on gradient L2-norm.
    2. **Attention-force dampening** – detects runaway gravitational collapse
       via per-layer force diagnostics and reduces those layers' G values.
    3. **Bekenstein entropy penalty** – adds a regularisation term that
       penalises high-entropy representations (information compression).
    4. **Divergence / collapse detection** – halts training if loss exceeds
       bounds or drops suspiciously low.
    5. **Architecture proposals** – suggests adding layers or increasing
       curvature when the model is sufficiently stable.

    Args:
        config: ``ContainmentConfig`` instance.
        model: The LGT model being trained.
        ledger: Optional Ledger for logging containment events.
    """

    def __init__(
        self,
        config: ContainmentConfig,
        model: nn.Module,
        ledger: Optional[Any] = None,  # victorcos_module.Ledger
    ):
        self.config = config
        self.model = model
        self.ledger = ledger
        self._stability_ema: float = 1.0
        self._step: int = 0
        self._proposals_made: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-step hook (call after backward, before optimizer.step)
    # ------------------------------------------------------------------

    def step(
        self,
        loss: torch.Tensor,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply all containment checks for the current training step.

        Returns a summary dict with keys: ``clipped``, ``damped``,
        ``stopped`` (bool), ``stability``, ``proposal``.
        """
        self._step += 1
        summary: Dict[str, Any] = {
            "step": self._step,
            "loss": loss.item(),
            "clipped": False,
            "damped": False,
            "stopped": False,
            "stability": self._stability_ema,
            "proposal": None,
        }

        # 1. Divergence / collapse detection
        loss_val = loss.item()
        if math.isnan(loss_val) or loss_val > self.config.max_loss:
            summary["stopped"] = True
            self._log("containment_stop", {"reason": "divergence", "loss": loss_val})
            return summary
        if loss_val < self.config.min_loss:
            summary["stopped"] = True
            self._log("containment_stop", {"reason": "collapse", "loss": loss_val})
            return summary

        # 2. Gradient clipping
        total_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        if float(total_norm) > self.config.max_grad_norm:
            summary["clipped"] = True
            self._log("grad_clip", {"total_norm": float(total_norm)})

        # 3. Attention-force dampening via diagnostics
        if diagnostics is not None:
            mean_force = self._extract_mean_force(diagnostics)
            self._update_stability(mean_force)
            summary["stability"] = self._stability_ema

            if mean_force > self.config.max_attention_force:
                self._damp_gravitational_constants()
                summary["damped"] = True
                self._log("attention_dampening", {
                    "mean_force": mean_force,
                    "threshold": self.config.max_attention_force,
                })

        # 4. Architecture proposal
        if (
            self.config.enable_architecture_proposals
            and self._stability_ema >= self.config.stability_proposal_threshold
        ):
            proposal = self._make_proposal()
            if proposal:
                summary["proposal"] = proposal

        return summary

    # ------------------------------------------------------------------
    # Bekenstein entropy regularisation term
    # ------------------------------------------------------------------

    def bekenstein_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Bekenstein-inspired entropy regularisation term.

        Penalises high-variance (high-entropy) layer outputs, encouraging
        the model to compress information into a low-dimensional manifold.

        The penalty is proportional to the mean per-token entropy estimated
        via the normalised variance of the representation:

            H ≈ 0.5 * log(2π e · var(x))  (Gaussian entropy upper bound)

        Args:
            x: Layer output tensor of any shape ``[..., dim]``.

        Returns:
            Scalar penalty tensor.
        """
        var = x.var(dim=-1).clamp(min=1e-8)
        entropy_upper_bound = 0.5 * torch.log(2.0 * math.pi * math.e * var)
        return self.config.bekenstein_lambda * entropy_upper_bound.mean()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_mean_force(self, diagnostics: Dict[str, Any]) -> float:
        layers = diagnostics.get("layers") or []
        if not layers:
            return 0.0
        forces = [layer.get("mean_force", 0.0) for layer in layers]
        return sum(forces) / len(forces) if forces else 0.0

    def _update_stability(self, mean_force: float) -> None:
        # High force → low stability; use EMA over normalised score
        alpha = self.config.stability_ema_alpha
        score = 1.0 / (1.0 + mean_force / max(self.config.max_attention_force, 1e-9))
        self._stability_ema = (1 - alpha) * self._stability_ema + alpha * score

    def _damp_gravitational_constants(self) -> None:
        """Reduce G for all attention heads by 10% to cool runaway dynamics."""
        for module in self.model.modules():
            if hasattr(module, "G") and isinstance(module.G, nn.Parameter):
                with torch.no_grad():
                    module.G.mul_(0.9)

    def _make_proposal(self) -> Optional[Dict[str, Any]]:
        """Generate an architecture mutation proposal (throttled by ``proposal_min_interval``)."""
        min_interval = self.config.proposal_min_interval
        if self._proposals_made and self._step - self._proposals_made[-1].get("step", 0) < min_interval:
            return None

        num_layers = sum(1 for m in self.model.modules() if m.__class__.__name__ == "LightweightGravitationalBlock")
        proposal: Dict[str, Any] = {
            "step": self._step,
            "stability": self._stability_ema,
        }
        if self._stability_ema > 0.98:
            proposal.update({"change": "add_layer", "new_num_layers": num_layers + 1})
        else:
            curvature = getattr(self.model, "curvature", 0.15)
            proposal.update({
                "change": "increase_curvature",
                "new_curvature": round(min(curvature * 1.1, 0.5), 4),
            })

        self._proposals_made.append(proposal)
        self._log("architecture_proposal", proposal)
        return proposal

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        if self.ledger is not None:
            self.ledger.log(event, payload)


# ---------------------------------------------------------------------------
# Meta-curvature scheduler
# ---------------------------------------------------------------------------

class MetaCurvatureScheduler:
    """
    Adjusts the learnable curvature parameters across LGT layers using a
    light meta-gradient signal.

    After each validation step the scheduler nudges the per-layer curvature
    in the direction that reduces validation loss, subject to the hard bounds
    ``[min_curvature, max_curvature]``.  This implements Phase 1 of the
    self-evolution pathway (parameter adaptation).

    Args:
        model: LGT model whose curvature parameters are adapted.
        lr: Meta-learning rate for curvature updates.
        min_curvature: Hard lower bound on curvature.
        max_curvature: Hard upper bound on curvature.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        min_curvature: float = 0.0,
        max_curvature: float = 0.5,
    ):
        self.model = model
        self.lr = lr
        self.min_curvature = min_curvature
        self.max_curvature = max_curvature
        self._prev_val_loss: Optional[float] = None

    def step(self, val_loss: float) -> Dict[str, float]:
        """
        Update curvature parameters based on change in validation loss.

        Returns a dict mapping parameter name → new curvature value.
        """
        updates: Dict[str, float] = {}
        if self._prev_val_loss is None:
            self._prev_val_loss = val_loss
            return updates

        delta = val_loss - self._prev_val_loss  # positive = loss increased

        for name, param in self.model.named_parameters():
            if "curvature" in name:
                with torch.no_grad():
                    # If loss increased, reduce curvature; if decreased, grow it
                    adjustment = -self.lr * math.copysign(1.0, delta)
                    param.add_(adjustment)
                    param.clamp_(self.min_curvature, self.max_curvature)
                    updates[name] = param.item()

        self._prev_val_loss = val_loss
        return updates


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for the LGT training loop."""
    max_steps: int = 10_000
    eval_every: int = 500
    log_every: int = 50
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    use_bekenstein_penalty: bool = True
    use_meta_curvature: bool = True
    meta_curvature_lr: float = 0.01
    grad_accumulation_steps: int = 1


class TrainingLoop:
    """
    Full training loop for the Lightweight Gravitational Transformer.

    Features:
    - ContainmentProtocol for gradient clipping, force dampening, and
      divergence detection at every step.
    - Optional Bekenstein entropy penalty added to the base loss.
    - MetaCurvatureScheduler for self-evolving positional geometry.
    - Mirror Layer integration for real-time cognitive state logging.
    - Ledger logging for full causal traceability.
    - Architecture mutation proposals emitted to the caller.

    Args:
        model: LGT model.
        optimizer: Torch optimiser.
        loss_fn: Callable ``(logits, targets) → scalar loss``.
        config: ``TrainingConfig`` instance.
        containment_config: ``ContainmentConfig`` instance.
        ledger: Optional Ledger for event logging.
        scheduler: Optional LR scheduler.
        device: Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: Optional[TrainingConfig] = None,
        containment_config: Optional[ContainmentConfig] = None,
        ledger: Optional[Any] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config or TrainingConfig()
        self.ledger = ledger
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")

        self.containment = ContainmentProtocol(
            config=containment_config or ContainmentConfig(),
            model=model,
            ledger=ledger,
        )

        self.meta_curvature: Optional[MetaCurvatureScheduler] = (
            MetaCurvatureScheduler(model, lr=self.config.meta_curvature_lr)
            if self.config.use_meta_curvature
            else None
        )

        self._step = 0
        self._train_losses: List[float] = []
        self._proposals: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a single training step.

        Args:
            batch: ``(inputs, targets)`` tuple.
            return_diagnostics: Forward with LGT diagnostics (slower).

        Returns:
            Dict with ``loss``, ``grad_norm``, ``stopped``, ``proposal``.
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.model.train()

        # Forward pass
        logits, diagnostics = self.model(
            inputs, return_diagnostics=return_diagnostics
        )

        # Compute loss
        loss = self.loss_fn(logits, targets)

        # Bekenstein penalty
        if self.config.use_bekenstein_penalty and diagnostics is not None:
            loss = loss + self.containment.bekenstein_penalty(logits)

        # Backward
        loss = loss / self.config.grad_accumulation_steps
        loss.backward()

        result: Dict[str, Any] = {"loss": loss.item() * self.config.grad_accumulation_steps}

        if (self._step + 1) % self.config.grad_accumulation_steps == 0:
            containment_summary = self.containment.step(loss * self.config.grad_accumulation_steps, diagnostics)
            result.update({
                "grad_norm_clipped": containment_summary["clipped"],
                "attention_damped": containment_summary["damped"],
                "stopped": containment_summary["stopped"],
                "stability": containment_summary["stability"],
                "proposal": containment_summary.get("proposal"),
            })
            if containment_summary.get("proposal"):
                self._proposals.append(containment_summary["proposal"])

            if not containment_summary["stopped"]:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            result.update({"stopped": False, "stability": self.containment._stability_ema})

        self._step += 1
        self._train_losses.append(result["loss"])

        if self.ledger and self._step % self.config.log_every == 0:
            self.ledger.log("train_step", {
                "step": self._step,
                "loss": result["loss"],
                "stability": result.get("stability", 1.0),
            })

        return result

    def eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """Run a single evaluation step and return the loss."""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(inputs, return_diagnostics=False)
            loss = self.loss_fn(logits, targets)

        val_loss = loss.item()

        if self.meta_curvature is not None:
            updates = self.meta_curvature.step(val_loss)
            if updates and self.ledger:
                self.ledger.log("meta_curvature_update", {"updates": updates})

        return val_loss

    def fit(
        self,
        train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        val_iter: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None,
        on_proposal: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.

        Args:
            train_iter: Iterator that yields ``(inputs, targets)`` batches.
            val_iter: Optional validation iterator (evaluated every
                ``config.eval_every`` steps).
            on_proposal: Optional callback invoked when an architecture
                mutation proposal is generated.

        Returns:
            Summary dict with ``steps``, ``final_loss``, ``proposals``.
        """
        self.optimizer.zero_grad()
        final_loss = float("inf")

        for step_idx, batch in enumerate(train_iter):
            if step_idx >= self.config.max_steps:
                break

            result = self.train_step(batch, return_diagnostics=(step_idx % 10 == 0))
            final_loss = result["loss"]

            if result.get("stopped"):
                if self.ledger:
                    self.ledger.log("training_stopped", {"step": self._step, "loss": final_loss})
                break

            if result.get("proposal") and on_proposal:
                on_proposal(result["proposal"])

            if val_iter is not None and step_idx % self.config.eval_every == 0:
                try:
                    val_batch = next(val_iter)  # type: ignore[call-overload]
                    val_loss = self.eval_step(val_batch)
                    if self.ledger:
                        self.ledger.log("eval_step", {"step": self._step, "val_loss": val_loss})
                except StopIteration:
                    pass

        return {
            "steps": self._step,
            "final_loss": final_loss,
            "proposals": self._proposals,
        }

    @property
    def proposals(self) -> List[Dict[str, Any]]:
        """All architecture mutation proposals generated so far."""
        return list(self._proposals)
