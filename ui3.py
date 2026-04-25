import math
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import numpy as np

from algorithms.benchmarks import ObjectiveFn, PRESET_CONFIG
from algorithms.gmo import GMO
from ui_sections import (
    ExperimentSetupSection,
    GraphsSection,
    LiveMetricsSection,
    RunLogSection,
)

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
        self.style.configure("Metric.TLabel", font=("Segoe UI", 15, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))

    def _initialize_state(self) -> None:
        self.gmo: GMO | None = None
        self.is_running = False
        self.run_after_id = None
        self.run_start_time = None

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

        self.live_metrics = LiveMetricsSection(right_panel)
        self.graphs = GraphsSection(right_panel)

        self.experiment_setup = ExperimentSetupSection(
            parent=left_panel,
            on_initialize=self.initialize_run,
            on_step=self.step_run,
            on_run=self.start_run,
            on_preset_changed=self._on_preset_changed,
            default_population=DEFAULT_POPULATION_SIZE,
            default_iterations=DEFAULT_ITERATIONS,
            default_custom_expression=DEFAULT_CUSTOM_EXPRESSION,
        )
        self.run_log = RunLogSection(left_panel)

        self.graphs.clear()

    def _on_preset_changed(self) -> None:
        self.graphs.reset_landscape_cache()

    # ---------------------------------------------------------
    # INPUT HELPERS
    # ---------------------------------------------------------
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
        values = self.experiment_setup.get_values()
        preset = values["preset"]
        function_expression = values["function_expression"]
        dim = int(values["dim"])
        population_size = int(values["population_size"])
        iterations = int(values["iterations"])
        seed = int(DEFAULT_SEED)

        if dim <= 0:
            raise ValueError("Dimensions must be positive.")
        if population_size < 3:
            raise ValueError("Population size must be at least 3.")
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")

        bounds = self._parse_bounds(values["bounds"], dim)
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
        self.run_log.append(message)

    def _show_solution_summary(self) -> None:
        if self.gmo is None:
            return

        best_x, best_f, best_v = self.gmo.best_solution()
        self._append_log("--- RUN COMPLETE ---")
        self._append_log(f"Best objective: {best_f:.8f}")
        self._append_log(f"Best violation: {best_v:.8f}")
        self._append_log(f"Best x: {self._format_vector(best_x)}")

    def _update_metrics(self) -> None:
        if self.gmo is None:
            self.live_metrics.set_defaults()
            return

        metrics = self.gmo.get_last_metrics()
        if metrics is None:
            _best_x, best_f, _best_v = self.gmo.best_solution()
            self.live_metrics.set_initial(self.gmo.max_iters, best_f)
            return

        elapsed = 0.0
        if self.run_start_time is not None:
            elapsed = time.time() - self.run_start_time
        self.live_metrics.set_runtime(metrics, self.gmo.max_iters, elapsed)

    def refresh_view(self) -> None:
        if self.gmo is None:
            self.live_metrics.set_defaults()
            self.graphs.clear()
            return

        history = self.gmo.get_history()
        self.graphs.refresh(self.gmo, history)
        self._update_metrics()

    # ---------------------------------------------------------
    # RUN CONTROLS
    # ---------------------------------------------------------
    def initialize_run(self) -> None:
        try:
            if self.run_after_id is not None:
                self.root.after_cancel(self.run_after_id)
                self.run_after_id = None
            self.is_running = False
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
            self.graphs.reset_landscape_cache()

            self.run_log.clear()
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
            self._append_log("Run already completed. Use Initialize.")
            return

        if self.is_running:
            return

        self.is_running = True
        self._append_log("Run started.")
        self._run_loop()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()