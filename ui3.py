import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import time

# Keep your original import
from algorithms.gmo import GMO


# =========================
# TAB 1 OBJECTIVE FUNCTION
# =========================
def simple_parabola(x):
    # f(x) = x^2, with 0 constraint violations
    return x[0]**2, 0


def sphere_objective(x):
    return sum(v * v for v in x), 0


def rosenbrock_objective(x):
    if len(x) < 2:
        return sphere_objective(x)
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2
    return total, 0


def rastrigin_objective(x):
    n = len(x)
    total = 10.0 * n
    for v in x:
        total += v * v - 10.0 * math.cos(2.0 * math.pi * v)
    return total, 0


def ackley_objective(x):
    n = len(x)
    if n == 0:
        return 0.0, 0

    sum_sq = sum(v * v for v in x)
    sum_cos = sum(math.cos(2.0 * math.pi * v) for v in x)

    term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / n))
    term2 = -math.exp(sum_cos / n)
    return term1 + term2 + 20.0 + math.e, 0


PRESET_CONFIG = {
    "Sphere": {
        "dim": 2,
        "bounds": "-5.12,5.12;-5.12,5.12",
        "expression": "sum(v*v for v in x)",
        "fn": sphere_objective,
    },
    "Rosenbrock": {
        "dim": 2,
        "bounds": "-3,3;-3,3",
        "expression": "sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))",
        "fn": rosenbrock_objective,
    },
    "Rastrigin": {
        "dim": 2,
        "bounds": "-5.12,5.12;-5.12,5.12",
        "expression": "10*len(x) + sum(v*v - 10*math.cos(2*math.pi*v) for v in x)",
        "fn": rastrigin_objective,
    },
    "Ackley": {
        "dim": 2,
        "bounds": "-32.768,32.768;-32.768,32.768",
        "expression": "-20*exp(-0.2*sqrt(sum(v*v for v in x)/len(x))) - exp(sum(cos(2*pi*v) for v in x)/len(x)) + 20 + e",
        "fn": ackley_objective,
    },
}


# =========================
# UI DASHBOARD
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("GMO Optimization Dashboard")
        self.root.geometry("1240x840")

        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        self.style.configure("Body.TLabel", font=("Segoe UI", 10))
        self.style.configure("Metric.TLabel", font=("Segoe UI", 10, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))

        self.gmo_tab2 = None
        self.tab2_running = False
        self.tab2_after_id = None
        self.tab2_start_time = None
        self.tab2_problem_name = ""

        # Create Tab Control
        self.tabControl = ttk.Notebook(root)
        
        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        
        self.tabControl.add(self.tab1, text='Step-by-Step (x²)')
        self.tabControl.add(self.tab2, text='Custom Function Minimizer')
        self.tabControl.pack(expand=1, fill="both")
        
        # Initialize both tabs
        self.init_tab1()
        self.init_tab2()

    # ---------------------------------------------------------
    # TAB 1: STEP-BY-STEP VISUALIZATION
    # ---------------------------------------------------------
    def init_tab1(self):
        # Control Frame
        control_frame = tk.Frame(self.tab1)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        tk.Button(control_frame, text="Initialize / Reset GMO", command=self.reset_gmo_tab1).pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="Next Step", command=self.step_gmo_tab1).pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="Run 10 Steps", command=lambda: [self.step_gmo_tab1() for _ in range(10)]).pack(side=tk.LEFT, padx=10)
        
        self.iter_label = tk.Label(control_frame, text="Iteration: 0", font=("Arial", 12, "bold"))
        self.iter_label.pack(side=tk.LEFT, padx=20)

        # Matplotlib Plot Frame
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 5))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.gmo_tab1 = None
        self.reset_gmo_tab1()

    def reset_gmo_tab1(self):
        # Setup GMO for a 1D x^2 problem
        self.gmo_tab1 = GMO(dim=1, pop_size=15, iters=100, bounds=[(-10, 10)], obj_fn=simple_parabola)
        self.update_plot_tab1()

    def step_gmo_tab1(self):
        if self.gmo_tab1 and self.gmo_tab1.current_iteration < self.gmo_tab1.max_iters:
            self.gmo_tab1.step()
            self.update_plot_tab1()

    def update_plot_tab1(self):
        self.ax1.clear()
        
        # Plot objective function landscape
        x_vals = np.linspace(-10, 10, 200)
        y_vals = x_vals**2
        self.ax1.plot(x_vals, y_vals, label="f(x) = x²", color="blue", alpha=0.3, linewidth=2)
        
        # Plot agents
        if self.gmo_tab1:
            agent_x = [a.x[0] for a in self.gmo_tab1.agents]
            agent_y = [a.x[0]**2 for a in self.gmo_tab1.agents]
            self.ax1.scatter(agent_x, agent_y, color="red", zorder=5, label="Search Agents")
            
            # Highlight global best
            best_x, best_f, _ = self.gmo_tab1.best_solution()
            self.ax1.scatter(best_x[0], best_f, color="gold", edgecolors="black", s=150, zorder=10, label="Best Solution", marker="*")
            
            self.iter_label.config(text=f"Iteration: {self.gmo_tab1.current_iteration}")

        self.ax1.set_title("1D GMO Optimization on x²")
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("f(x)")
        self.ax1.legend()
        self.canvas1.draw()

    # ---------------------------------------------------------
    # TAB 2: CUSTOM FUNCTION MINIMIZER
    # ---------------------------------------------------------
    def init_tab2(self):
        container = ttk.Frame(self.tab2, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container, width=360)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        right = ttk.Frame(container)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(left, text="Experiment Setup", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="Preset:", style="Body.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        self.preset_var = tk.StringVar(value="Sphere")
        self.preset_combo = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=["Sphere", "Rosenbrock", "Rastrigin", "Ackley", "Custom"],
            width=24,
            state="readonly",
        )
        self.preset_combo.grid(row=0, column=1, sticky="w", pady=4)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_changed)

        ttk.Label(controls, text="Objective Function f(x):", style="Body.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        self.func_entry = tk.Entry(controls, width=40)
        self.func_entry.grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Dimensions:", style="Body.TLabel").grid(row=2, column=0, sticky="w", pady=4)
        self.dim_entry = ttk.Entry(controls, width=12)
        self.dim_entry.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Bounds (min,max;...):", style="Body.TLabel").grid(row=3, column=0, sticky="w", pady=4)
        self.bounds_entry = ttk.Entry(controls, width=40)
        self.bounds_entry.grid(row=3, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Population Size:", style="Body.TLabel").grid(row=4, column=0, sticky="w", pady=4)
        self.pop_entry = ttk.Entry(controls, width=12)
        self.pop_entry.grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Iterations:", style="Body.TLabel").grid(row=5, column=0, sticky="w", pady=4)
        self.iter_entry = ttk.Entry(controls, width=12)
        self.iter_entry.grid(row=5, column=1, sticky="w", pady=4)

        ttk.Label(controls, text="Seed:", style="Body.TLabel").grid(row=6, column=0, sticky="w", pady=4)
        self.seed_entry = ttk.Entry(controls, width=12)
        self.seed_entry.grid(row=6, column=1, sticky="w", pady=4)

        btn_row1 = ttk.Frame(controls)
        btn_row1.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Button(btn_row1, text="Initialize", command=self.initialize_custom, style="Action.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row1, text="Step", command=self.step_custom, style="Action.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row1, text="Run", command=self.run_custom, style="Action.TButton").pack(side=tk.LEFT, padx=6)

        btn_row2 = ttk.Frame(controls)
        btn_row2.grid(row=8, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(btn_row2, text="Pause", command=self.pause_custom).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row2, text="Resume", command=self.resume_custom).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row2, text="Reset", command=self.reset_custom).pack(side=tk.LEFT, padx=6)

        log_frame = ttk.LabelFrame(left, text="Run Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.output2 = tk.Text(log_frame, height=18, font=("Courier New", 11), wrap="word")
        self.output2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.output2.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output2.config(yscrollcommand=scroll.set)

        metrics_frame = ttk.LabelFrame(right, text="Live Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        self.metric_iteration = tk.StringVar(value="Iteration: 0/0")
        self.metric_best = tk.StringVar(value="Best Objective: -")
        self.metric_violation = tk.StringVar(value="Best Violation: -")
        self.metric_feasible = tk.StringVar(value="Feasible Agents: -")
        self.metric_stats = tk.StringVar(value="mu/sigma: -")
        self.metric_elites = tk.StringVar(value="Elites: -")
        self.metric_diversity = tk.StringVar(value="Diversity: -")
        self.metric_elapsed = tk.StringVar(value="Elapsed: 0.00s")

        metric_labels = [
            self.metric_iteration,
            self.metric_best,
            self.metric_violation,
            self.metric_feasible,
            self.metric_stats,
            self.metric_elites,
            self.metric_diversity,
            self.metric_elapsed,
        ]

        for i, var in enumerate(metric_labels):
            ttk.Label(metrics_frame, textvariable=var, style="Metric.TLabel").grid(
                row=i // 4,
                column=i % 4,
                sticky="w",
                padx=12,
                pady=3,
            )

        plot_frame = ttk.Frame(right)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig2, (self.ax2_top, self.ax2_bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        self.fig2.tight_layout(pad=2.0)

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plot_frame)
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.on_preset_changed()
        self._clear_tab2_plot()

    def on_preset_changed(self, _event=None):
        preset = self.preset_var.get()

        self.func_entry.config(state="normal")
        if preset == "Custom":
            if not self.func_entry.get().strip():
                self.func_entry.insert(0, "x[0]**2 + x[1]**2")
            return

        cfg = PRESET_CONFIG[preset]

        self.func_entry.delete(0, tk.END)
        self.func_entry.insert(0, cfg["expression"])
        self.func_entry.config(state="disabled")

        self.dim_entry.delete(0, tk.END)
        self.dim_entry.insert(0, str(cfg["dim"]))

        self.bounds_entry.delete(0, tk.END)
        self.bounds_entry.insert(0, cfg["bounds"])

        if not self.pop_entry.get().strip():
            self.pop_entry.insert(0, "30")
        if not self.iter_entry.get().strip():
            self.iter_entry.insert(0, "120")
        if not self.seed_entry.get().strip():
            self.seed_entry.insert(0, "123456")

    def _parse_bounds(self, bounds_text, dim):
        tokens = [t.strip() for t in bounds_text.split(";") if t.strip()]
        if len(tokens) == 1 and dim > 1:
            tokens = tokens * dim

        if len(tokens) != dim:
            raise ValueError(
                f"Expected {dim} bounds but got {len(tokens)}. Use min,max;min,max;..."
            )

        bounds = []
        for token in tokens:
            parts = [p.strip() for p in token.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Invalid bound format: '{token}'")
            low = float(parts[0])
            high = float(parts[1])
            if low >= high:
                raise ValueError(f"Each bound must satisfy min < max. Invalid: {token}")
            bounds.append((low, high))

        return bounds

    def _build_objective_function(self, preset, func_str):
        if preset != "Custom":
            return PRESET_CONFIG[preset]["fn"], preset

        def custom_obj(x):
            allowed_locals = {
                "x": x,
                "math": math,
                "np": np,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "len": len,
            }
            value = eval(func_str, {"__builtins__": None}, allowed_locals)
            return float(value), 0

        return custom_obj, "Custom"

    def initialize_custom(self):
        try:
            self.pause_custom(silent=True)

            preset = self.preset_var.get()
            func_str = self.func_entry.get().strip()
            dim = int(self.dim_entry.get())
            pop_size = int(self.pop_entry.get())
            iters = int(self.iter_entry.get())
            seed = int(self.seed_entry.get())

            if dim <= 0:
                raise ValueError("Dimensions must be positive.")
            if pop_size < 3:
                raise ValueError("Population size must be at least 3.")
            if iters <= 0:
                raise ValueError("Iterations must be positive.")

            bounds = self._parse_bounds(self.bounds_entry.get(), dim)
            obj_fn, problem_name = self._build_objective_function(preset, func_str)

            self.gmo_tab2 = GMO(
                dim=dim,
                pop_size=pop_size,
                iters=iters,
                bounds=bounds,
                obj_fn=obj_fn,
                seed=seed,
                track_history=True,
            )

            self.tab2_problem_name = problem_name
            self.tab2_start_time = time.time()

            self.output2.delete(1.0, tk.END)
            self._append_tab2_log(f"Initialized run: {problem_name}")
            self._append_tab2_log(
                f"dim={dim}, pop={pop_size}, iters={iters}, seed={seed}"
            )
            self._append_tab2_log(f"bounds={bounds}")

            self.refresh_tab2_view()

        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to initialize.\n\nDetails: {str(e)}")

    def _format_vector(self, values, limit=6):
        if not values:
            return "[]"
        shown = [f"{v:.4f}" for v in values[:limit]]
        if len(values) > limit:
            shown.append("...")
        return "[" + ", ".join(shown) + "]"

    def _append_tab2_log(self, msg):
        self.output2.insert(tk.END, msg + "\n")
        self.output2.see(tk.END)

    def _show_tab2_solution_summary(self):
        if not self.gmo_tab2:
            return

        best_x, best_f, best_v = self.gmo_tab2.best_solution()
        self._append_tab2_log("--- RUN COMPLETE ---")
        self._append_tab2_log(f"Best objective: {best_f:.8f}")
        self._append_tab2_log(f"Best violation: {best_v:.8f}")
        self._append_tab2_log(f"Best x: {self._format_vector(best_x)}")

    def _clear_tab2_plot(self):
        self.ax2_top.clear()
        self.ax2_bottom.clear()

        self.ax2_top.set_title("Convergence (Best Objective)")
        self.ax2_top.set_ylabel("Best Cost")
        self.ax2_top.grid(True, linestyle="--", alpha=0.4)

        self.ax2_bottom.set_title("Feasibility and Exploration")
        self.ax2_bottom.set_xlabel("Iteration")
        self.ax2_bottom.set_ylabel("Percent")
        self.ax2_bottom.set_ylim(0, 105)
        self.ax2_bottom.grid(True, linestyle="--", alpha=0.4)

        self.canvas2.draw()

    def refresh_tab2_view(self):
        if not self.gmo_tab2:
            self.metric_iteration.set("Iteration: 0/0")
            self.metric_best.set("Best Objective: -")
            self.metric_violation.set("Best Violation: -")
            self.metric_feasible.set("Feasible Agents: -")
            self.metric_stats.set("mu/sigma: -")
            self.metric_elites.set("Elites: -")
            self.metric_diversity.set("Diversity: -")
            self.metric_elapsed.set("Elapsed: 0.00s")
            self._clear_tab2_plot()
            return

        history = self.gmo_tab2.get_history()
        self.ax2_top.clear()
        self.ax2_bottom.clear()

        if history:
            iterations = [h["iteration"] for h in history]
            best_cost = [h["best_f"] for h in history]
            feasible_pct = [100.0 * h["feasible_ratio"] for h in history]
            exploration_pct = [100.0 * h["w"] for h in history]

            self.ax2_top.plot(iterations, best_cost, color="#2a9d8f", linewidth=2.0)
            self.ax2_bottom.plot(iterations, feasible_pct, color="#e76f51", linewidth=2.0, label="Feasible %")
            self.ax2_bottom.plot(iterations, exploration_pct, color="#264653", linewidth=1.8, label="Exploration Weight x100")
            self.ax2_bottom.legend(loc="upper right")

        self.ax2_top.set_title("Convergence (Best Objective)")
        self.ax2_top.set_ylabel("Best Cost")
        self.ax2_top.grid(True, linestyle="--", alpha=0.4)

        self.ax2_bottom.set_title("Feasibility and Exploration")
        self.ax2_bottom.set_xlabel("Iteration")
        self.ax2_bottom.set_ylabel("Percent")
        self.ax2_bottom.set_ylim(0, 105)
        self.ax2_bottom.grid(True, linestyle="--", alpha=0.4)

        metrics = self.gmo_tab2.get_last_metrics()
        if metrics is None:
            best_x, best_f, best_v = self.gmo_tab2.best_solution()
            feasible_count = sum(1 for a in self.gmo_tab2.agents if a.violation_best == 0)
            self.metric_iteration.set(f"Iteration: 0/{self.gmo_tab2.max_iters}")
            self.metric_best.set(f"Best Objective: {best_f:.8f}")
            self.metric_violation.set(f"Best Violation: {best_v:.8f}")
            self.metric_feasible.set(f"Feasible Agents: {feasible_count}/{self.gmo_tab2.pop_size}")
            self.metric_stats.set("mu/sigma: -")
            self.metric_elites.set("Elites: -")
            self.metric_diversity.set("Diversity: -")
            self.metric_elapsed.set("Elapsed: 0.00s")
        else:
            elapsed = 0.0
            if self.tab2_start_time is not None:
                elapsed = time.time() - self.tab2_start_time

            self.metric_iteration.set(
                f"Iteration: {metrics['iteration']}/{self.gmo_tab2.max_iters}"
            )
            self.metric_best.set(f"Best Objective: {metrics['best_f']:.8f}")
            self.metric_violation.set(f"Best Violation: {metrics['best_violation']:.8f}")
            self.metric_feasible.set(
                f"Feasible Agents: {metrics['feasible_count']}/{self.gmo_tab2.pop_size}"
            )
            self.metric_stats.set(
                f"mu/sigma: {metrics['mu']:.5f}/{metrics['sigma']:.5f}"
            )
            self.metric_elites.set(f"Elites: {metrics['elite_count']}")
            self.metric_diversity.set(
                f"Diversity: {metrics['diversity']:.5f} | mean|v|: {metrics['mean_abs_velocity']:.5f}"
            )
            self.metric_elapsed.set(f"Elapsed: {elapsed:.2f}s")

        self.canvas2.draw()

    def step_custom(self):
        if self.gmo_tab2 is None:
            self.initialize_custom()
            if self.gmo_tab2 is None:
                return

        if self.tab2_running:
            return

        progressed = self.gmo_tab2.step()
        self.refresh_tab2_view()

        if progressed:
            m = self.gmo_tab2.get_last_metrics()
            self._append_tab2_log(
                f"step {m['iteration']:>4}: best={m['best_f']:.7f}, feasible={m['feasible_count']}/{self.gmo_tab2.pop_size}, diversity={m['diversity']:.5f}"
            )
            if m["iteration"] >= self.gmo_tab2.max_iters:
                self._show_tab2_solution_summary()
        else:
            self._append_tab2_log("Reached max iterations.")
            self._show_tab2_solution_summary()

    def _run_custom_loop(self):
        if not self.tab2_running or self.gmo_tab2 is None:
            return

        steps_per_tick = 5
        did_step = False

        for _ in range(steps_per_tick):
            if not self.gmo_tab2.step():
                self.tab2_running = False
                break
            did_step = True

        if did_step:
            self.refresh_tab2_view()
            m = self.gmo_tab2.get_last_metrics()
            if m and m["iteration"] % 10 == 0:
                self._append_tab2_log(
                    f"iter {m['iteration']:>4}: best={m['best_f']:.7f}, feasible={m['feasible_count']}/{self.gmo_tab2.pop_size}"
                )

        if self.tab2_running:
            self.tab2_after_id = self.root.after(30, self._run_custom_loop)
        else:
            self.refresh_tab2_view()
            self._append_tab2_log("Run completed.")
            self._show_tab2_solution_summary()

    def run_custom(self):
        if self.gmo_tab2 is None:
            self.initialize_custom()
            if self.gmo_tab2 is None:
                return

        if self.gmo_tab2.current_iteration >= self.gmo_tab2.max_iters:
            self._append_tab2_log("Run already completed. Use Reset or Initialize.")
            return

        if self.tab2_running:
            return

        self.tab2_running = True
        self._append_tab2_log("Run started.")
        self._run_custom_loop()

    def pause_custom(self, silent=False):
        if self.tab2_after_id is not None:
            self.root.after_cancel(self.tab2_after_id)
            self.tab2_after_id = None

        was_running = self.tab2_running
        self.tab2_running = False

        if was_running and not silent:
            self._append_tab2_log("Run paused.")

    def resume_custom(self):
        if self.gmo_tab2 is None:
            self._append_tab2_log("Initialize first.")
            return

        if self.gmo_tab2.current_iteration >= self.gmo_tab2.max_iters:
            self._append_tab2_log("Run already completed.")
            return

        if self.tab2_running:
            return

        self.tab2_running = True
        self._append_tab2_log("Run resumed.")
        self._run_custom_loop()

    def reset_custom(self):
        self.pause_custom(silent=True)
        self.gmo_tab2 = None
        self.tab2_start_time = None
        self.output2.delete(1.0, tk.END)
        self._append_tab2_log("State reset. Configure and initialize a new run.")
        self.refresh_tab2_view()


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()