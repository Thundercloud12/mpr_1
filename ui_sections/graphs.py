import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphsSection:
    def __init__(self, parent: tk.Widget) -> None:
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = plt.figure(figsize=(8, 9.5))
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.6, 1.0])
        self.ax_obj = self.figure.add_subplot(grid[0, 0])
        self.ax_convergence = self.figure.add_subplot(grid[1, 0])
        self.figure.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.landscape_cache = None

    def reset_landscape_cache(self) -> None:
        self.landscape_cache = None

    def clear(self) -> None:
        self.ax_obj.clear()
        self.ax_convergence.clear()

        self.ax_obj.set_title("Objective Function and Agents")
        self.ax_obj.set_xlabel("x[0]")
        self.ax_obj.set_ylabel("f(x)")
        self.ax_obj.grid(True, linestyle="--", alpha=0.35)

        self.ax_convergence.set_title("Convergence (Best Objective)")
        self.ax_convergence.set_xlabel("Iteration")
        self.ax_convergence.set_ylabel("Best Cost")
        self.ax_convergence.grid(True, linestyle="--", alpha=0.4)

        self.canvas.draw()

    def _safe_objective_value(self, gmo, point) -> float:
        if gmo is None:
            return np.nan

        try:
            value, _ = gmo.obj_fn(point)
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                return np.nan
            return numeric_value
        except Exception:
            return np.nan

    def _build_landscape_cache(self, gmo):
        if gmo is None:
            return None

        low, high = gmo.bounds[0]
        x_values = np.linspace(low, high, 260)

        base_point = []
        for bound_low, bound_high in gmo.bounds:
            base_point.append((bound_low + bound_high) * 0.5)

        y_values = []
        for x in x_values:
            point = base_point[:]
            point[0] = x
            y_values.append(self._safe_objective_value(gmo, point))

        return {"x": x_values, "y": np.array(y_values, dtype=float)}

    def _draw_objective_plot(self, gmo, history) -> None:
        self.ax_obj.clear()

        if gmo is None:
            self.ax_obj.set_title("Objective Function and Agents")
            self.ax_obj.text(
                0.5,
                0.5,
                "Initialize run to display objective graph",
                transform=self.ax_obj.transAxes,
                ha="center",
                va="center",
            )
            self.ax_obj.set_xticks([])
            self.ax_obj.set_yticks([])
            return

        if self.landscape_cache is None:
            self.landscape_cache = self._build_landscape_cache(gmo)

        if self.landscape_cache is None:
            self.ax_obj.set_title("Objective Function and Agents")
            self.ax_obj.text(
                0.5,
                0.5,
                "Unable to evaluate objective landscape",
                transform=self.ax_obj.transAxes,
                ha="center",
                va="center",
            )
            return

        x_values = self.landscape_cache["x"]
        y_values = self.landscape_cache["y"]
        current_best = history[-1] if history else None

        self.ax_obj.plot(
            x_values,
            y_values,
            color="#3a86ff",
            linewidth=2.0,
            alpha=0.85,
            label="Objective",
        )

        agent_x = [agent.x[0] for agent in gmo.agents]
        agent_y = [self._safe_objective_value(gmo, agent.x) for agent in gmo.agents]
        self.ax_obj.scatter(agent_x, agent_y, color="red", s=36, zorder=5, label="Agents")

        if current_best is not None:
            self.ax_obj.scatter(
                current_best["best_x"][0],
                current_best["best_f"],
                color="yellow",
                edgecolors="black",
                s=170,
                marker="*",
                zorder=8,
                label="Current Best So Far",
            )

        title = "Objective Function and Agents"
        if gmo.dim > 1:
            title += " (x[0] slice)"
        self.ax_obj.set_title(title)
        self.ax_obj.set_xlabel("x[0]")
        self.ax_obj.set_ylabel("f(x)")
        self.ax_obj.grid(True, linestyle="--", alpha=0.35)
        self.ax_obj.legend(loc="upper right")

        if current_best is None:
            self.ax_obj.text(
                0.02,
                0.95,
                "Best star appears after first optimization step",
                transform=self.ax_obj.transAxes,
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#f8f9fa",
                    "edgecolor": "#cfd8dc",
                },
            )

    def _draw_progress_plot(self, history) -> None:
        self.ax_convergence.clear()

        if history:
            iterations = [item["iteration"] for item in history]
            best_cost = [item["best_f"] for item in history]

            self.ax_convergence.plot(iterations, best_cost, color="#2a9d8f", linewidth=2.0)

        self.ax_convergence.set_title("Convergence (Best Objective)")
        self.ax_convergence.set_xlabel("Iteration")
        self.ax_convergence.set_ylabel("Best Cost")
        self.ax_convergence.grid(True, linestyle="--", alpha=0.4)

    def refresh(self, gmo, history) -> None:
        self._draw_objective_plot(gmo, history)
        self._draw_progress_plot(history)
        self.canvas.draw()
