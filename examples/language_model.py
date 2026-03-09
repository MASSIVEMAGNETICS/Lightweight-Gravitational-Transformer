"""
Language Model Training Example
================================
Demonstrates a minimal language-model training loop using LGT with the
ContainmentProtocol, Bekenstein penalty, and Ledger integration.

Run from the repository root:
    python examples/language_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from lightweight_gravitational_transformer import LightweightGravitationalTransformer
from training import TrainingLoop, TrainingConfig, ContainmentConfig
from victorcos_module import Ledger


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def synthetic_data(vocab_size: int = 500, seq_len: int = 16, batch_size: int = 8):
    """Infinite iterator yielding (input_ids, target_ids) batches."""
    while True:
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Shift-by-one target (next-token prediction)
        y = torch.roll(x, shifts=-1, dims=1)
        yield x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    VOCAB_SIZE = 500
    DIM_MODEL  = 64
    NUM_LAYERS = 2
    NUM_HEADS  = 2
    MAX_STEPS  = 200

    print("Building model …")
    model = LightweightGravitationalTransformer(
        vocab_size=VOCAB_SIZE,
        dim_model=DIM_MODEL,
        dim_position=32,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=64,
        curvature=0.15,
        dropout=0.1,
        tie_weights=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")

    # Ledger (memory-only for this example)
    ledger = Ledger(agent_id="lm_example")

    # Loss function: flatten logits and targets for CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss()

    def flat_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return loss_fn(logits.view(-1, VOCAB_SIZE), targets.view(-1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        loss_fn=flat_loss,
        config=TrainingConfig(
            max_steps=MAX_STEPS,
            eval_every=50,
            log_every=25,
            use_bekenstein_penalty=True,
            use_meta_curvature=True,
        ),
        containment_config=ContainmentConfig(
            max_grad_norm=1.0,
            max_attention_force=40.0,
            bekenstein_lambda=1e-4,
        ),
        ledger=ledger,
    )

    proposals_received = []

    def on_proposal(proposal):
        proposals_received.append(proposal)
        print(f"  [proposal] {proposal}")

    print(f"\nTraining for {MAX_STEPS} steps …")
    train_iter = synthetic_data(VOCAB_SIZE)
    val_iter   = synthetic_data(VOCAB_SIZE)

    summary = loop.fit(train_iter, val_iter=val_iter, on_proposal=on_proposal)

    print(f"\nTraining complete:")
    print(f"  Steps      : {summary['steps']}")
    print(f"  Final loss : {summary['final_loss']:.4f}")
    print(f"  Proposals  : {len(proposals_received)}")
    print(f"  Ledger entries: {len(ledger)}")

    # Show some ledger events
    train_events = ledger.entries(event_filter="train_step")
    if train_events:
        last = train_events[-1]
        print(f"\nLast train_step log:")
        print(f"  step={last.payload['step']}, loss={last.payload['loss']:.4f}, "
              f"stability={last.payload['stability']:.3f}")

    print("\nLanguage model training example completed ✓")


if __name__ == "__main__":
    main()
