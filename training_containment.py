"""
Morphic Containment Protocol
Physics-aware stabiliser for high-gravity (Singularity) training phases.

When the gravitational constant ``G`` spikes to ``50.0`` during the
Singularity phase, standard back-propagation can produce very large or NaN
gradients because attention weights become extremely peaked.

The ``MorphicContainmentProtocol`` wraps each optimiser step and:

1. **Bekenstein Entropy penalty** – penalises *information collapse* (near-zero
   Shannon entropy in the attention distribution), preventing the model from
   collapsing into a permanently singular state.
2. **Hawking Radiation (force-aware gradient damping)** – when the recorded
   ``max_force`` from the forward pass exceeds ``max_attention_force``, all
   parameter gradients are scaled down proportionally.
3. **Global gradient clipping** – a hard cap on the total gradient L2-norm.
4. **Stability check** – returns ``False`` to signal the training loop to
   halt / reset when the stability score drops below ``min_stability``.
5. **Ledger integration** – every containment event (damping, breach) is
   logged to the VictorOS :class:`~victorcos_module.Ledger`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MorphicContainmentConfig:
    """
    Configuration for the :class:`MorphicContainmentProtocol`.

    Attributes:
        max_grad_norm: Maximum L2-norm of gradients; triggers global clipping.
        max_attention_force: Cap for the maximum gravitational force recorded
            by the forward pass.  Exceeding this triggers Hawking-Radiation
            gradient damping.
        bekenstein_lambda: Regularisation weight for the Bekenstein Entropy
            penalty.  A higher value penalises information collapse more
            aggressively.
        min_stability: Emergency shutdown threshold.  When the stability
            score falls below this value, :meth:`MorphicContainmentProtocol.step`
            returns ``False`` to signal the caller to halt or reset.
        entropy_target: Target mean Shannon entropy for the attention
            distribution.  The penalty is zero when the distribution entropy
            equals this value and grows as entropy deviates downward.
    """
    max_grad_norm: float = 1.0
    max_attention_force: float = 100.0
    bekenstein_lambda: float = 1e-4
    min_stability: float = 0.2
    entropy_target: float = 1.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class MorphicContainmentProtocol:
    """
    Ensures the Polymorphic Attention Orchestrator does not implode during
    high-gravity (Singularity) phases.

    Args:
        model: The model whose parameters are guarded.
        ledger: Optional VictorOS :class:`~victorcos_module.Ledger` for
            logging containment events.
        config: A :class:`MorphicContainmentConfig` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        ledger: Optional[Any] = None,  # victorcos_module.Ledger
        config: Optional[MorphicContainmentConfig] = None,
    ):
        self.model = model
        self.ledger = ledger
        self.config = config if config is not None else MorphicContainmentConfig()

    # ------------------------------------------------------------------
    # Bekenstein entropy penalty
    # ------------------------------------------------------------------

    def apply_bekenstein_penalty(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute the Bekenstein Entropy penalty for a given attention distribution.

        The Bekenstein Bound states that information in a spatial region is
        limited by its surface area.  Here we penalise *information collapse*:
        if the attention distribution has very low Shannon entropy, the model
        is "seeing" only one or two tokens (a singularity), which corresponds
        to extreme information compression.

        The penalty is::

            λ · (1 / (H + ε))²

        where ``H`` is the mean Shannon entropy of the attention distribution
        and ``ε`` prevents division-by-zero.

        Args:
            attention_weights: Attention distribution tensor of shape
                ``[..., seq_len]`` (last dimension must sum to 1, i.e. after
                softmax).

        Returns:
            Scalar penalty tensor (differentiable).
        """
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights.clamp(min=1e-9)),
            dim=-1,
        )  # [...] – one scalar per query position
        mean_entropy = entropy.mean()
        penalty = self.config.bekenstein_lambda * (
            1.0 / (mean_entropy + 1e-6)
        ) ** 2
        return penalty

    # ------------------------------------------------------------------
    # Per-step containment hook
    # ------------------------------------------------------------------

    def step(
        self,
        loss: torch.Tensor,
        model_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Apply all containment checks for the current training step.

        Call this **after** ``loss.backward()`` and **before**
        ``optimizer.step()``.

        Args:
            loss: The scalar training loss (used only for Ledger logging).
            model_diagnostics: Optional dict from the forward pass containing
                at least ``"max_force"`` and ``"phase"`` keys (as returned by
                :meth:`~polymorphic_attention_orchestrator.PolymorphicAttentionOrchestrator.forward`).

        Returns:
            ``True`` if training should continue; ``False`` if the caller
            should halt or reset (stability below ``min_stability``).
        """
        diagnostics = model_diagnostics or {}

        # 1. Evaluate force intensity and apply Hawking Radiation if needed
        max_f = float(diagnostics.get("max_force", 0.0))
        phase = str(diagnostics.get("phase", "fluid"))

        if max_f > self.config.max_attention_force:
            damping_factor = self.config.max_attention_force / max_f
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(damping_factor)

            self._log("containment_event", {
                "type": "hawking_radiation",
                "phase": phase,
                "max_force": max_f,
                "damping_factor": damping_factor,
                "loss": loss.item(),
            })

        # 2. Global gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        # 3. Stability check
        stability = float(diagnostics.get("stability", 1.0))
        if stability < self.config.min_stability:
            self._log("containment_breach", {
                "type": "stability_collapse",
                "stability": stability,
                "threshold": self.config.min_stability,
                "phase": phase,
                "loss": loss.item(),
            })
            return False  # Signal caller to halt / reset

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        """Forward a log entry to the attached Ledger (if any)."""
        if self.ledger is not None:
            self.ledger.log(event, payload)
