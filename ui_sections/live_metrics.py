from collections.abc import Mapping

import tkinter as tk
from tkinter import ttk


class LiveMetricsSection:
    def __init__(self, parent: ttk.Frame) -> None:
        metrics_frame = ttk.LabelFrame(parent, text="Live Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)

        self.metric_vars = {
            "iteration": tk.StringVar(),
            "best": tk.StringVar(),
            "diversity": tk.StringVar(),
            "elapsed": tk.StringVar(),
        }

        metric_order = ["iteration", "best", "diversity", "elapsed"]
        for idx, metric_key in enumerate(metric_order):
            ttk.Label(
                metrics_frame,
                textvariable=self.metric_vars[metric_key],
                style="Metric.TLabel",
            ).grid(row=idx // 2, column=idx % 2, sticky="w", padx=12, pady=6)

        self.set_defaults()

    def set_defaults(self) -> None:
        self.metric_vars["iteration"].set("Iteration: 0/0")
        self.metric_vars["best"].set("Best Objective: -")
        self.metric_vars["diversity"].set("Diversity: -")
        self.metric_vars["elapsed"].set("Elapsed: 0.00s")

    def set_initial(self, max_iters: int, best_f: float) -> None:
        self.metric_vars["iteration"].set(f"Iteration: 0/{max_iters}")
        self.metric_vars["best"].set(f"Best Objective: {best_f:.8f}")
        self.metric_vars["diversity"].set("Diversity: -")
        self.metric_vars["elapsed"].set("Elapsed: 0.00s")

    def set_runtime(self, metrics: Mapping[str, float], max_iters: int, elapsed: float) -> None:
        self.metric_vars["iteration"].set(
            f"Iteration: {int(metrics['iteration'])}/{max_iters}"
        )
        self.metric_vars["best"].set(f"Best Objective: {float(metrics['best_f']):.8f}")
        self.metric_vars["diversity"].set(
            f"Diversity: {float(metrics['diversity']):.5f}"
        )
        self.metric_vars["elapsed"].set(f"Elapsed: {elapsed:.2f}s")
