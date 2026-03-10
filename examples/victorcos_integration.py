"""
VictorOS Integration Example
=============================
Demonstrates the Ledger, MirrorLayer, LGTVictorOSModule, and the
@victoros_module decorator.

Run from the repository root:
    python examples/victorcos_integration.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer
from victorcos_module import (
    Ledger,
    MirrorLayer,
    LGTVictorOSModule,
    VictorOSBaseModule,
    victoros_module,
)


# ---------------------------------------------------------------------------
# Example 1: Ledger basics
# ---------------------------------------------------------------------------

def ledger_example():
    print("=" * 60)
    print("Example 1: Ledger")
    print("=" * 60)

    ledger = Ledger(agent_id="demo_agent")

    # Log arbitrary structured events
    ledger.log("startup", {"version": "0.1.0"})
    ledger.log("inference", {"seq_len": 32, "output_norm": 1.23})
    ledger.log("inference", {"seq_len": 16, "output_norm": 0.87})
    ledger.log("checkpoint", {"path": "ckpt_step100.pt"})

    print(f"Total entries    : {len(ledger)}")
    print(f"Inference entries: {len(ledger.entries(event_filter='inference'))}")

    snapshot = ledger.snapshot()
    print(f"Snapshot keys    : {list(snapshot.keys())}")

    # Show entries
    for entry in ledger.entries():
        print(f"  [{entry.event}] {entry.payload}")


# ---------------------------------------------------------------------------
# Example 2: MirrorLayer
# ---------------------------------------------------------------------------

def mirror_layer_example():
    print()
    print("=" * 60)
    print("Example 2: MirrorLayer")
    print("=" * 60)

    model = LightweightGravitationalTransformer(
        dim_model=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    model.eval()

    ledger = Ledger(agent_id="mirror_demo")
    corrections_received = []

    mirror = MirrorLayer(
        ledger=ledger,
        max_force_threshold=40.0,
        stability_window=10,
        correction_callback=lambda layer_idx, correction_type: corrections_received.append(
            {"layer": layer_idx, "correction": correction_type}
        ),
    )

    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        output, _ = model(x, return_diagnostics=True, mirror_layer_callback=mirror)

    print(f"Stability score     : {mirror.stability_score():.4f}")
    print(f"Corrections         : {corrections_received}")
    print(f"Mirror ledger events: {len(ledger)}")

    # The mirror layer logs "mirror_layer" events
    mirror_events = ledger.entries(event_filter="mirror_layer")
    if mirror_events:
        ev = mirror_events[0]
        print(f"First mirror event  : layer={ev.payload['layer']}, "
              f"stability={ev.payload['stability_score']:.4f}")


# ---------------------------------------------------------------------------
# Example 3: LGTVictorOSModule
# ---------------------------------------------------------------------------

def lgt_victorcos_module_example():
    print()
    print("=" * 60)
    print("Example 3: LGTVictorOSModule")
    print("=" * 60)

    model = LightweightGravitationalTransformer(
        dim_model=64,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )

    module = LGTVictorOSModule(
        model=model,
        agent_id="lgt_core_demo",
        persist_path=None,           # memory-only Ledger
        max_force_threshold=40.0,
    )

    x = torch.randn(2, 16, 64)
    result = module.process(x, return_diagnostics=True)

    print(f"Output shape  : {result['output'].shape}")
    print(f"Stability     : {result['stability']:.4f}")
    print(f"Ledger entries: {len(module.ledger)}")

    # Attention snapshot
    snapshot = module.get_snapshot(x[:1])
    print(f"Snapshot config: {snapshot['model_config']}")

    # Architecture proposal (may be None if stability is too low)
    proposal = module.propose_architecture_change(
        current_config={"num_layers": 2, "curvature": 0.15},
        stability_threshold=0.0,   # always propose for demo purposes
    )
    if proposal:
        print(f"Proposal: {proposal}")


# ---------------------------------------------------------------------------
# Example 4: @victoros_module decorator
# ---------------------------------------------------------------------------

def custom_module_example():
    print()
    print("=" * 60)
    print("Example 4: @victoros_module Decorator")
    print("=" * 60)

    @victoros_module(
        name="custom_lgt_agent",
        version="1.0.0",
        requirements=["torch>=2.0.0"],
        containment_native=True,
        description="Custom LGT cognitive module for demonstration.",
    )
    class CustomAgent(VictorOSBaseModule):
        def __init__(self, dim_model: int = 64):
            # @victoros_module wraps __init__ to auto-provision ledger + mirror_layer
            self.model = LightweightGravitationalTransformer(
                dim_model=dim_model,
                num_layers=2,
                num_heads=2,
                dropout=0.0,
            )

        def process(self, x: torch.Tensor) -> torch.Tensor:
            self.model.eval()
            with torch.no_grad():
                output, _ = self.model(
                    x,
                    return_diagnostics=True,
                    mirror_layer_callback=self.mirror_layer,
                )
            self.ledger.log("inference", {
                "shape": list(x.shape),
                "stability": self.mirror_layer.stability_score(),
                "output_norm": float(output.norm()),
            })
            return output

    agent = CustomAgent(dim_model=64)

    print(f"Module name     : {agent._victoros_meta.name}")
    print(f"Module version  : {agent._victoros_meta.version}")
    print(f"Containment     : {agent._victoros_meta.containment_native}")

    x = torch.randn(1, 8, 64)
    output = agent.process(x)
    print(f"Output shape    : {output.shape}")
    print(f"Ledger entries  : {len(agent.ledger)}")

    inference_events = agent.ledger.entries(event_filter="inference")
    if inference_events:
        ev = inference_events[0]
        print(f"Inference log   : stability={ev.payload['stability']:.4f}, "
              f"norm={ev.payload['output_norm']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ledger_example()
    mirror_layer_example()
    lgt_victorcos_module_example()
    custom_module_example()
    print("\nAll VictorOS integration examples completed ✓")
