import math
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.benchmarks import ObjectiveFn, PRESET_CONFIG, PRESET_NAMES
from algorithms.gmo import GMO

RUN_LOOP_DELAY_MS = 30
RUN_LOOP_STEPS_PER_TICK = 5
DEFAULT_POPULATION_SIZE = "30"
DEFAULT_ITERATIONS = "120"
DEFAULT_SEED = "123456"
DEFAULT_CUSTOM_EXPRESSION = "x[0]**2 + x[1]**2"


@dataclass(frozen=True)
class RunConfig:
    preset: str
    function_expression: str
    dim: int
    bounds: list[tuple[float, float]]
    population_size: int
    iterations: int
    seed: int


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._configure_window()
        self._configure_styles()
        self._initialize_state()
        self._build_layout()

    # ---------------------------------------------------------
    # APP INITIALIZATION
    # ---------------------------------------------------------
    def _configure_window(self) -> None:
        self.root.title("GMO Optimization Dashboard")
        self.root.geometry("1240x840")

    def _configure_styles(self) -> None:
        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        self.style.configure("Body.TLabel", font=("Segoe UI", 10))
        self.style.configure("Metric.TLabel", font=("Segoe UI", 10, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))

    def _initialize_state(self) -> None:
        self.gmo: GMO | None = None
        self.is_running = False
        self.run_after_id = None
        self.run_start_time = None
        self.landscape_cache = None

    def _build_layout(self) -> None:
        self.tab_control = ttk.Notebook(self.root)
        self.main_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.main_tab, text="Custom Function Minimizer")
        self.tab_control.pack(expand=1, fill="both")

        self._build_main_tab()

    def _build_main_tab(self) -> None:
        container = ttk.Frame(self.main_tab, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(container, width=360)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls_panel(left_panel)
        self._build_log_panel(left_panel)
        self._build_metrics_panel(right_panel)
        self._build_plot_panel(right_panel)

        self.on_preset_changed()
        self._clear_plots()

    # ---------------------------------------------------------
    # UI BUILDERS
    # ---------------------------------------------------------
    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Experiment Setup", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Preset:", style="Body.TLabel").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.preset_var = tk.StringVar(value="Sphere")
        self.preset_combo = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=PRESET_NAMES,
            width=24,
            state="readonly",
        )
        self.preset_combo.grid(row=0, column=1, sticky="w", pady=4)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_changed)

        ttk.Label(controls, text="Objective Function f(x):", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.func_entry = ttk.Entry(controls, width=40)
        self.func_entry.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Dimensions:", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.dim_entry = ttk.Entry(controls, width=12)
        self.dim_entry.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Bounds (min,max;...):", style="Body.TLabel").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.bounds_entry = ttk.Entry(controls, width=40)
        self.bounds_entry.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Population Size:", style="Body.TLabel").grid(
            row=4, column=0, sticky="w", pady=4
        )
        self.pop_entry = ttk.Entry(controls, width=12)
        self.pop_entry.grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Iterations:", style="Body.TLabel").grid(
            row=5, column=0, sticky="w", pady=4
        )
        self.iter_entry = ttk.Entry(controls, width=12)
        self.iter_entry.grid(row=5, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Seed:", style="Body.TLabel").grid(
            row=6, column=0, sticky="w", pady=4
        )
        self.seed_entry = ttk.Entry(controls, width=12)
        self.seed_entry.grid(row=6, column=1, sticky="w", pady=4)

        run_buttons = ttk.Frame(controls)
        run_buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Button(
            run_buttons,
            text="Initialize",
            command=self.initialize_run,
            style="Action.TButton",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            run_buttons,
            text="Step",
            command=self.step_run,
            style="Action.TButton",
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            run_buttons,
            text="Run",
            command=self.start_run,
            style="Action.TButton",
        ).pack(side=tk.LEFT, padx=6)

        transport_buttons = ttk.Frame(controls)
        transport_buttons.grid(row=8, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(transport_buttons, text="Pause", command=self.pause_run).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(transport_buttons, text="Resume", command=self.resume_run).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(transport_buttons, text="Reset", command=self.reset_run).pack(
            side=tk.LEFT, padx=6
        )

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        log_frame = ttk.LabelFrame(parent, text="Run Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_output = tk.Text(log_frame, height=18, font=("Courier New", 11), wrap="word")
        self.log_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_output.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_output.config(yscrollcommand=scrollbar.set)

    def _build_metrics_panel(self, parent: ttk.Frame) -> None:
        metrics_frame = ttk.LabelFrame(parent, text="Live Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        self.metric_vars = {
            "iteration": tk.StringVar(),
            "best": tk.StringVar(),
            "violation": tk.StringVar(),
            "feasible": tk.StringVar(),
            "stats": tk.StringVar(),
            "elites": tk.StringVar(),
            "diversity": tk.StringVar(),
            "elapsed": tk.StringVar(),
        }
        self._set_metric_defaults()

        metric_order = [
            "iteration",
            "best",
            "violation",
            "feasible",
            "stats",
            "elites",
            "diversity",
            "elapsed",
        ]

        for idx, metric_key in enumerate(metric_order):
            ttk.Label(
                metrics_frame,
                textvariable=self.metric_vars[metric_key],
                style="Metric.TLabel",
            ).grid(row=idx // 4, column=idx % 4, sticky="w", padx=12, pady=3)

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = plt.figure(figsize=(8, 9.5))
        grid = self.figure.add_gridspec(3, 1, height_ratios=[1.45, 1.0, 1.0])
        self.ax_obj = self.figure.add_subplot(grid[0, 0])
        self.ax_convergence = self.figure.add_subplot(grid[1, 0])
        self.ax_feasibility = self.figure.add_subplot(grid[2, 0], sharex=self.ax_convergence)
        self.figure.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ---------------------------------------------------------
    # INPUT HELPERS
    # ---------------------------------------------------------
    def _set_entry_if_empty(self, entry_widget: ttk.Entry, value: str) -> None:
        if not entry_widget.get().strip():
            entry_widget.insert(0, value)

    def _replace_entry_text(self, entry_widget: ttk.Entry, value: str) -> None:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, value)

    def on_preset_changed(self, _event=None) -> None:
        preset = self.preset_var.get()
        self.landscape_cache = None
        self.func_entry.config(state="normal")

        if preset == "Custom":
            self._set_entry_if_empty(self.func_entry, DEFAULT_CUSTOM_EXPRESSION)
            self._set_entry_if_empty(self.dim_entry, "2")
            self._set_entry_if_empty(self.bounds_entry, "-5.12,5.12;-5.12,5.12")
        else:
            cfg = PRESET_CONFIG[preset]
            self._replace_entry_text(self.func_entry, cfg.expression)
            self.func_entry.config(state="disabled")
            self._replace_entry_text(self.dim_entry, str(cfg.dim))
            self._replace_entry_text(self.bounds_entry, cfg.bounds)

        self._set_entry_if_empty(self.pop_entry, DEFAULT_POPULATION_SIZE)
        self._set_entry_if_empty(self.iter_entry, DEFAULT_ITERATIONS)
        self._set_entry_if_empty(self.seed_entry, DEFAULT_SEED)

    def _parse_bounds(self, bounds_text: str, dim: int) -> list[tuple[float, float]]:
        tokens = [token.strip() for token in bounds_text.split(";") if token.strip()]
        if len(tokens) == 1 and dim > 1:
            tokens *= dim

        if len(tokens) != dim:
            raise ValueError(
                f"Expected {dim} bounds but got {len(tokens)}. Use min,max;min,max;..."
            )

        parsed_bounds = []
        for token in tokens:
            parts = [part.strip() for part in token.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Invalid bound format: '{token}'")

            low, high = float(parts[0]), float(parts[1])
            if low >= high:
                raise ValueError(f"Each bound must satisfy min < max. Invalid: {token}")
            parsed_bounds.append((low, high))

        return parsed_bounds

    def _read_run_config(self) -> RunConfig:
        preset = self.preset_var.get()
        function_expression = self.func_entry.get().strip()
        dim = int(self.dim_entry.get())
        population_size = int(self.pop_entry.get())
        iterations = int(self.iter_entry.get())
        seed = int(self.seed_entry.get())

        if dim <= 0:
            raise ValueError("Dimensions must be positive.")
        if population_size < 3:
            raise ValueError("Population size must be at least 3.")
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")

        bounds = self._parse_bounds(self.bounds_entry.get(), dim)
        return RunConfig(
            preset=preset,
            function_expression=function_expression,
            dim=dim,
            bounds=bounds,
            population_size=population_size,
            iterations=iterations,
            seed=seed,
        )

    def _build_objective_function(
        self, preset: str, function_expression: str
    ) -> tuple[ObjectiveFn, str]:
        if preset != "Custom":
            return PRESET_CONFIG[preset].fn, preset

        def custom_objective(x):
            allowed_locals = {
                "x": x,
                "math": math,
                "np": np,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "len": len,
                "pow": pow,
                "round": round,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "exp": math.exp,
                "log": math.log,
                "pi": math.pi,
                "e": math.e,
            }
            value = eval(function_expression, {"__builtins__": None}, allowed_locals)
            return float(value), 0

        return custom_objective, "Custom"

    # ---------------------------------------------------------
    # LOGGING + METRICS
    # ---------------------------------------------------------
    def _format_vector(self, values, limit: int = 6) -> str:
        if not values:
            return "[]"
        shown = [f"{v:.4f}" for v in values[:limit]]
        if len(values) > limit:
            shown.append("...")
        return "[" + ", ".join(shown) + "]"

    def _append_log(self, message: str) -> None:
        self.log_output.insert(tk.END, message + "\n")
        self.log_output.see(tk.END)

    def _show_solution_summary(self) -> None:
        if self.gmo is None:
            return

        best_x, best_f, best_v = self.gmo.best_solution()
        self._append_log("--- RUN COMPLETE ---")
        self._append_log(f"Best objective: {best_f:.8f}")
        self._append_log(f"Best violation: {best_v:.8f}")
        self._append_log(f"Best x: {self._format_vector(best_x)}")

    def _set_metric_defaults(self) -> None:
        self.metric_vars["iteration"].set("Iteration: 0/0")
        self.metric_vars["best"].set("Best Objective: -")
        self.metric_vars["violation"].set("Best Violation: -")
        self.metric_vars["feasible"].set("Feasible Agents: -")
        self.metric_vars["stats"].set("mu/sigma: -")
        self.metric_vars["elites"].set("Elites: -")
        self.metric_vars["diversity"].set("Diversity: -")
        self.metric_vars["elapsed"].set("Elapsed: 0.00s")

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    def _clear_plots(self) -> None:
        self.ax_obj.clear()
        self.ax_convergence.clear()
        self.ax_feasibility.clear()

        self.ax_obj.set_title("Objective Function and Agents")
        self.ax_obj.set_xlabel("x[0]")
        self.ax_obj.set_ylabel("f(x)")
        self.ax_obj.grid(True, linestyle="--", alpha=0.35)

        self.ax_convergence.set_title("Convergence (Best Objective)")
        self.ax_convergence.set_ylabel("Best Cost")
        self.ax_convergence.grid(True, linestyle="--", alpha=0.4)

        self.ax_feasibility.set_title("Feasibility and Exploration")
        self.ax_feasibility.set_xlabel("Iteration")
        self.ax_feasibility.set_ylabel("Percent")
        self.ax_feasibility.set_ylim(0, 105)
        self.ax_feasibility.grid(True, linestyle="--", alpha=0.4)

        self.canvas.draw()

    def _safe_objective_value(self, point) -> float:
        if self.gmo is None:
            return np.nan

        try:
            value, _ = self.gmo.obj_fn(point)
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                return np.nan
            return numeric_value
        except Exception:
            return np.nan

    def _build_landscape_cache(self):
        if self.gmo is None:
            return None

        low, high = self.gmo.bounds[0]
        x_values = np.linspace(low, high, 260)

        base_point = []
        for bound_low, bound_high in self.gmo.bounds:
            base_point.append((bound_low + bound_high) * 0.5)

        y_values = []
        for x in x_values:
            point = base_point[:]
            point[0] = x
            y_values.append(self._safe_objective_value(point))

        return {"x": x_values, "y": np.array(y_values, dtype=float)}

    def _draw_objective_plot(self, history) -> None:
        self.ax_obj.clear()

        if self.gmo is None:
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
            self.landscape_cache = self._build_landscape_cache()

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

        agent_x = [agent.x[0] for agent in self.gmo.agents]
        agent_y = [self._safe_objective_value(agent.x) for agent in self.gmo.agents]
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
        if self.gmo.dim > 1:
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

    def _draw_progress_plots(self, history) -> None:
        self.ax_convergence.clear()
        self.ax_feasibility.clear()

        if history:
            iterations = [item["iteration"] for item in history]
            best_cost = [item["best_f"] for item in history]
            feasible_percent = [100.0 * item["feasible_ratio"] for item in history]
            exploration_percent = [100.0 * item["w"] for item in history]

            self.ax_convergence.plot(iterations, best_cost, color="#2a9d8f", linewidth=2.0)
            self.ax_feasibility.plot(
                iterations,
                feasible_percent,
                color="#e76f51",
                linewidth=2.0,
                label="Feasible %",
            )
            self.ax_feasibility.plot(
                iterations,
                exploration_percent,
                color="#264653",
                linewidth=1.8,
                label="Exploration Weight x100",
            )
            self.ax_feasibility.legend(loc="upper right")

        self.ax_convergence.set_title("Convergence (Best Objective)")
        self.ax_convergence.set_ylabel("Best Cost")
        self.ax_convergence.grid(True, linestyle="--", alpha=0.4)

        self.ax_feasibility.set_title("Feasibility and Exploration")
        self.ax_feasibility.set_xlabel("Iteration")
        self.ax_feasibility.set_ylabel("Percent")
        self.ax_feasibility.set_ylim(0, 105)
        self.ax_feasibility.grid(True, linestyle="--", alpha=0.4)

    def _update_metrics(self) -> None:
        if self.gmo is None:
            self._set_metric_defaults()
            return

        metrics = self.gmo.get_last_metrics()
        if metrics is None:
            _best_x, best_f, best_v = self.gmo.best_solution()
            feasible_count = sum(
                1 for agent in self.gmo.agents if agent.violation_best == 0
            )

            self.metric_vars["iteration"].set(f"Iteration: 0/{self.gmo.max_iters}")
            self.metric_vars["best"].set(f"Best Objective: {best_f:.8f}")
            self.metric_vars["violation"].set(f"Best Violation: {best_v:.8f}")
            self.metric_vars["feasible"].set(
                f"Feasible Agents: {feasible_count}/{self.gmo.pop_size}"
            )
            self.metric_vars["stats"].set("mu/sigma: -")
            self.metric_vars["elites"].set("Elites: -")
            self.metric_vars["diversity"].set("Diversity: -")
            self.metric_vars["elapsed"].set("Elapsed: 0.00s")
            return

        elapsed = 0.0
        if self.run_start_time is not None:
            elapsed = time.time() - self.run_start_time

        self.metric_vars["iteration"].set(
            f"Iteration: {metrics['iteration']}/{self.gmo.max_iters}"
        )
        self.metric_vars["best"].set(f"Best Objective: {metrics['best_f']:.8f}")
        self.metric_vars["violation"].set(
            f"Best Violation: {metrics['best_violation']:.8f}"
        )
        self.metric_vars["feasible"].set(
            f"Feasible Agents: {metrics['feasible_count']}/{self.gmo.pop_size}"
        )
        self.metric_vars["stats"].set(
            f"mu/sigma: {metrics['mu']:.5f}/{metrics['sigma']:.5f}"
        )
        self.metric_vars["elites"].set(f"Elites: {metrics['elite_count']}")
        self.metric_vars["diversity"].set(
            f"Diversity: {metrics['diversity']:.5f} | mean|v|: {metrics['mean_abs_velocity']:.5f}"
        )
        self.metric_vars["elapsed"].set(f"Elapsed: {elapsed:.2f}s")

    def refresh_view(self) -> None:
        if self.gmo is None:
            self._set_metric_defaults()
            self._clear_plots()
            return

        history = self.gmo.get_history()
        self._draw_objective_plot(history)
        self._draw_progress_plots(history)
        self._update_metrics()
        self.canvas.draw()

    # ---------------------------------------------------------
    # RUN CONTROLS
    # ---------------------------------------------------------
    def initialize_run(self) -> None:
        try:
            self.pause_run(silent=True)
            config = self._read_run_config()
            objective_fn, problem_name = self._build_objective_function(
                config.preset, config.function_expression
            )

            self.gmo = GMO(
                dim=config.dim,
                pop_size=config.population_size,
                iters=config.iterations,
                bounds=config.bounds,
                obj_fn=objective_fn,
                seed=config.seed,
                track_history=True,
            )

            self.run_start_time = time.time()
            self.landscape_cache = None

            self.log_output.delete("1.0", tk.END)
            self._append_log(f"Initialized run: {problem_name}")
            self._append_log(
                f"dim={config.dim}, pop={config.population_size}, iters={config.iterations}, seed={config.seed}"
            )
            self._append_log(f"bounds={config.bounds}")

            self.refresh_view()
        except Exception as exc:
            messagebox.showerror(
                "Configuration Error", f"Failed to initialize.\n\nDetails: {exc}"
            )

    def step_run(self) -> None:
        if self.gmo is None:
            self.initialize_run()
            if self.gmo is None:
                return

        if self.is_running:
            return

        progressed = self.gmo.step()
        self.refresh_view()

        if progressed:
            metrics = self.gmo.get_last_metrics()
            if metrics is not None:
                self._append_log(
                    f"step {metrics['iteration']:>4}: best={metrics['best_f']:.7f}, feasible={metrics['feasible_count']}/{self.gmo.pop_size}, diversity={metrics['diversity']:.5f}"
                )
                if metrics["iteration"] >= self.gmo.max_iters:
                    self._show_solution_summary()
            return

        self._append_log("Reached max iterations.")
        self._show_solution_summary()

    def _run_loop(self) -> None:
        if not self.is_running or self.gmo is None:
            return

        did_step = False
        for _ in range(RUN_LOOP_STEPS_PER_TICK):
            if not self.gmo.step():
                self.is_running = False
                break
            did_step = True

        if did_step:
            self.refresh_view()
            metrics = self.gmo.get_last_metrics()
            if metrics is not None and metrics["iteration"] % 10 == 0:
                self._append_log(
                    f"iter {metrics['iteration']:>4}: best={metrics['best_f']:.7f}, feasible={metrics['feasible_count']}/{self.gmo.pop_size}"
                )

        if self.is_running:
            self.run_after_id = self.root.after(RUN_LOOP_DELAY_MS, self._run_loop)
            return

        self.run_after_id = None
        self.refresh_view()
        self._append_log("Run completed.")
        self._show_solution_summary()

    def start_run(self) -> None:
        if self.gmo is None:
            self.initialize_run()
            if self.gmo is None:
                return

        if self.gmo.current_iteration >= self.gmo.max_iters:
            self._append_log("Run already completed. Use Reset or Initialize.")
            return

        if self.is_running:
            return

        self.is_running = True
        self._append_log("Run started.")
        self._run_loop()

    def pause_run(self, silent: bool = False) -> None:
        if self.run_after_id is not None:
            self.root.after_cancel(self.run_after_id)
            self.run_after_id = None

        was_running = self.is_running
        self.is_running = False

        if was_running and not silent:
            self._append_log("Run paused.")

    def resume_run(self) -> None:
        if self.gmo is None:
            self._append_log("Initialize first.")
            return

        if self.gmo.current_iteration >= self.gmo.max_iters:
            self._append_log("Run already completed.")
            return

        if self.is_running:
            return

        self.is_running = True
        self._append_log("Run resumed.")
        self._run_loop()

    def reset_run(self) -> None:
        self.pause_run(silent=True)
        self.gmo = None
        self.run_start_time = None
        self.landscape_cache = None
        self.log_output.delete("1.0", tk.END)
        self._append_log("State reset. Configure and initialize a new run.")
        self.refresh_view()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()