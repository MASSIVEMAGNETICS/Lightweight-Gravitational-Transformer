#!/usr/bin/env python3
"""
boxol-flower – CLI entrypoint for the Boxol Flower simulation.

Usage::

    boxol-flower --gui
    boxol-flower --headless
    boxol-flower --headless --layers 4 --z-modulo 2 --input "hello"

If ``--gui`` is requested but tkinter / matplotlib are unavailable (e.g. on
a headless CI server), the tool falls back to ``--headless`` automatically
and logs a warning.

Default parameters match ``BoxolFlowerMonolith v1.6``:
    layers=6, spacing=3.5, z-modulo=3
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("boxol_flower.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boxol-flower",
        description=(
            "Boxol Flower v1.6 — Gravitational Chess Executive + "
            "4 Vassal Buses + 5-Tick CoT simulation."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Launch interactive Tkinter/Matplotlib GUI (requires display).",
    )
    mode.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run headless simulation and print voxel statistics.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        metavar="N",
        help="Number of hexagonal petal rings (default: 6).",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=3.5,
        metavar="F",
        help="Radial multiplier: ring r placed at r*spacing (default: 3.5).",
    )
    parser.add_argument(
        "--z-modulo",
        type=int,
        default=3,
        dest="z_modulo",
        metavar="N",
        help="Modulo for z-coordinate of each ring (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        metavar="N",
        help="Pendulum ticks for headless simulation (default: 5).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="TEXT",
        help="Sensory input text for the executive decision.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser


def _run_headless(args: argparse.Namespace) -> None:
    """Run a headless simulation and print a summary."""
    from boxol_flower import BoxolFlower  # noqa: PLC0415

    flower = BoxolFlower(
        layers=args.layers,
        spacing=args.spacing,
        z_modulo=args.z_modulo,
        seed=args.seed,
    )

    _log.info(
        "Headless simulation: total=%d sacred=%d petal=%d",
        flower.voxel_count,
        flower.sacred_count,
        flower.petal_count,
    )

    if args.input:
        flower.sensory_input(args.input)

    result = flower.pendulum_bloom(ticks=args.steps)
    _log.info("CoT: %s", result["cot"])

    if args.input:
        decision = flower.executive_decide()
        _log.info(
            "Executive decision: priority=%.4f risk=%.4f pulse=%d",
            decision["priority"],
            decision["risk"],
            decision["pulse"],
        )

    for bus_id in (1, 2, 3, 4):
        _log.debug(flower.vassal_bus(bus_id))

    data = flower.render()
    print(
        f"Boxol Flower stats: "
        f"total={flower.voxel_count} "
        f"sacred={flower.sacred_count} "
        f"petal={flower.petal_count} "
        f"active={int(data['active'].sum())}"
    )


def _run_gui(args: argparse.Namespace) -> None:  # pragma: no cover
    """Launch the Tkinter/Matplotlib GUI, falling back to headless on error."""
    from boxol_flower import BoxolFlower, GuiController  # noqa: PLC0415

    flower = BoxolFlower(
        layers=args.layers,
        spacing=args.spacing,
        z_modulo=args.z_modulo,
        seed=args.seed,
    )
    if args.input:
        flower.sensory_input(args.input)

    try:
        gui = GuiController(flower)
        gui.run()
    except Exception as exc:
        # Catches RuntimeError (GUI libs missing), TclError (no display),
        # and any other GUI initialisation failure.
        _log.warning(
            "GUI unavailable (%s). Falling back to headless mode.", exc
        )
        _run_headless(args)


def main(argv: Optional[list] = None) -> int:
    """Entry point; returns an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to headless when neither flag is given.
    if not args.gui and not args.headless:
        args.headless = True

    try:
        if args.gui:
            _run_gui(args)
        else:
            _run_headless(args)
    except KeyboardInterrupt:
        _log.info("Interrupted by user.")
    except Exception as exc:  # noqa: BLE001
        _log.error("Fatal error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
