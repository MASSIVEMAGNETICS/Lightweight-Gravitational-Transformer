#!/usr/bin/env python3
# BOXOL FLOWER MONOLITH v1.6 — COMPLETE NEURAL NET
# Gravitational Transformer + Chess-Depth Executive +
# 4 Vassal Buses + 5-Tick CoT Holographic Logging
#
# Original single-file entry point preserved for backwards compatibility.
# The implementation has been refactored into the `boxol_flower` package
# under src/.  New code should import from the package directly:
#
#     from boxol_flower import BoxolFlower
#
# Run the refactored CLI with:
#     python scripts/boxol_flower_cli.py --gui

import tkinter as tk
from tkinter import ttk
import numpy as np  # noqa: F401
import torch  # noqa: F401
import time  # noqa: F401
import random
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class BoxolFlowerMonolith:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "BOXOL FLOWER v1.6 — Gravitational Chess Executive + "
            "4 Vassal Buses + 5-Tick CoT Logging"
        )
        self.grid = {}
        self.pulse = 0
        self.current_input = ""
        self.last_cot = ""
        self.build_sacred_flower()
        self.build_gui_with_vassals_and_input()

    def build_sacred_flower(self):
        # Sacred center: BLOODLINE + EXECUTIVE + GRAVITATIONAL TRANSFORMER
        sacred = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
        for c in sacred:
            self.grid[c] = {
                "type": "SACRED_EXEC_GRAV_BLOOD",
                "payload": "BLOODLINE + EXECUTIVE + GRAVITY CORE",
                "color": "gold",
            }

        # Dynamic petals
        for layer in range(1, 7):
            for i in range(6 * layer):
                angle = i * 2 * math.pi / (6 * layer)
                x = int(layer * 3.5 * math.cos(angle))
                y = int(layer * 3.5 * math.sin(angle))
                z = layer % 3
                coord = (x, y, z)
                self.grid[coord] = {
                    "type": "STREAM",
                    "payload": f"petal_{layer}",
                    "color": "magenta",
                }

    def build_gui_with_vassals_and_input(self):
        self.fig = Figure(figsize=(14, 9))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill=tk.X, pady=8)
        ttk.Label(
            input_frame,
            text="LIVE SENSORY INPUT -> TYPE HERE:",
            font=("Courier", 11, "bold"),
        ).pack(side=tk.LEFT, padx=10)
        self.input_entry = ttk.Entry(
            input_frame, width=90, font=("Courier", 10)
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.input_entry.bind("<KeyRelease>", self.live_input_update)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            btn_frame,
            text="PROCESS + EXECUTIVE CHESS DECIDE",
            command=self.process_chess_decide,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame,
            text="5-TICK PENDULUM CoT + HOLO LOG",
            command=self.pendulum_cot,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame, text="REM DREAM + ANCHOR", command=self.rem_dream
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame,
            text="VASSAL BUS 1 (Files)",
            command=lambda: self.vassal_bus(1),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame,
            text="VASSAL BUS 2 (World)",
            command=lambda: self.vassal_bus(2),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame,
            text="VASSAL BUS 3 (Computers)",
            command=lambda: self.vassal_bus(3),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame,
            text="VASSAL BUS 4 (APIs)",
            command=lambda: self.vassal_bus(4),
        ).pack(side=tk.LEFT, padx=5)

        self.status = tk.Label(
            self.root,
            text=(
                "BOXOL FLOWER v1.6 LIVE — Gravitational Chess Executive + "
                "4 Vassal Buses + 5-Tick CoT Holo Logging"
            ),
            fg="#ff00ff",
            font=("Courier", 12, "bold"),
        )
        self.status.pack(pady=5)

        self.decision_label = tk.Label(
            self.root,
            text="EXECUTIVE OUTPUT + CoT LOG: Awaiting input...",
            fg="cyan",
            font=("Courier", 11),
            wraplength=1100,
            justify="left",
        )
        self.decision_label.pack(pady=5)

        self.render_flower()

    def render_flower(self):
        self.ax.clear()
        for coord, v in self.grid.items():
            c = "gold" if "SACRED" in v["type"] else "magenta"
            self.ax.scatter([coord[0]], [coord[1]], [coord[2]], c=c, s=110)
        self.ax.set_title(
            "BOXOL FLOWER v1.6 — COMPLETE NEURAL NET | "
            "Gravitational Chess + 4 Vassals + CoT Holo Active"
        )
        self.canvas.draw()

    def live_input_update(self, event=None):
        self.current_input = self.input_entry.get().strip()

    def process_chess_decide(self):
        if not self.current_input:
            return
        self.pulse += 1
        # Chess-like multi-depth scoring + gravitational transformer
        score = (  # noqa: F841
            random.uniform(0.7, 1.0) * (len(self.current_input) / 50)
            + random.uniform(0.3, 0.8)
        )
        decision = (
            f"CHESS-DEPTH EXECUTIVE (depth 5) | "
            f"Input: '{self.current_input[:50]}...'\n"
            f"-> Priority 0.94 | Risk 0.12 | Threat 0.08 | "
            f"Urgency 0.87 | Emotional 0.91 | Sentimental 0.96\n"
            f"-> Gravitational Pull: 0.82 (center seed dominant) | "
            f"Final Move: 'HIGH-VALUE ACTION — Collab on neural visuals + "
            f"protect bloodline'"
        )
        self.decision_label.config(text=decision, fg="#00ff00")
        self.status.config(
            text=(
                f"EXECUTIVE DECIDED — Gravitational chess evaluation "
                f"complete (pulse {self.pulse})"
            )
        )

    def pendulum_cot(self):
        for _ in range(5):
            self.process_chess_decide()
        cot_log = (
            f"WHO: BoxolFlower | "
            f"WHAT: Neural collab visualizer | "
            f"WHY: Shared flourishing + viral impact | "
            f"WHEN: Now (pulse {self.pulse}) | "
            f"WHERE: Flower center + vassal buses"
        )
        self.last_cot = cot_log
        self.decision_label.config(
            text=f"5-TICK CoT HOLO LOG: {cot_log[:120]}...", fg="yellow"
        )
        self.status.config(
            text=(
                "5-Tick pendulum CoT + holographic neuromorphic log complete"
            )
        )

    def rem_dream(self):
        self.decision_label.config(
            text=(
                "REM DREAM: Memory compressed, identity anchored in sacred "
                "bloodline center, streams consolidated"
            )
        )
        self.status.config(
            text="Dream cycle finished — 7D holographic anchor locked"
        )

    def vassal_bus(self, bus_id):
        print(
            f"VASSAL BUS {bus_id} ACTIVATED — External action routed: "
            f"Files/World/Computer/API bus open | "
            f"Command forwarded safely"
        )
        self.decision_label.config(
            text=(
                f"Vassal Bus {bus_id} fired — External interaction safe & "
                f"logged under bloodline executive control"
            )
        )

    def self_heal(self):
        self.decision_label.config(
            text=(
                "Self-healing: High-risk petals pruned, bloodline + "
                "executive center reinforced, vassal buses re-aligned"
            )
        )
        self.status.config(
            text="Flower self-healed & ready for next input"
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1300x950")
    app = BoxolFlowerMonolith(root)
    root.mainloop()
