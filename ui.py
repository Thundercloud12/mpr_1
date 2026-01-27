import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import math
from algorithms.gmo import GMO
from algorithms.pso import PSO
from algorithms.de import DE

# ==============================
# Predefined Objective Functions
# ==============================
def sphere(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    x = np.array(x)
    f = np.sum(x**2)
    violation = 0
    return f, violation

def rastrigin(x):
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))"""
    x = np.array(x)
    A = 10
    n = len(x)
    f = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    violation = 0
    return f, violation

def resource_allocation_objective(x):
    """
    Engineered resource allocation objective.
    Total resources scales with problem dimension for realistic scenarios.
    Returns (objective_value, constraint_violation)
    """
    x = np.array(x)
    dim = len(x)
    
    # Scale total resources with dimension: ~10% allocation per unit on average
    TOTAL_RESOURCE = dim * 0.1
    NOISE_STD = 0.15

    # ---- System metrics ----
    total_used = np.sum(x)
    imbalance = np.std(x)                 # instability proxy
    geometric_risk = np.prod(x + 1e-6)**(1 / dim)

    # ---- Soft penalties ----
    overuse_penalty = max(0, total_used - TOTAL_RESOURCE) ** 2
    imbalance_penalty = imbalance ** 2

    # ---- Noise (models demand / sensor uncertainty) ----
    noise = random.gauss(0, NOISE_STD)

    # ---- Final objective ----
    f = (
        -geometric_risk               # maximize balanced utilization
        + 0.8 * imbalance_penalty
        + 2.0 * overuse_penalty
        + noise
    )

    # Constraint violation channel (optional but explicit)
    violation = max(0, total_used - TOTAL_RESOURCE)

    return f, violation

def speed_reducer_objective(x):
    """
    Speed Reducer Design Optimization
    Minimize weight subject to 11 nonlinear constraints
    Returns (objective_value, constraint_violation)
    """

    x1, x2, x3, x4, x5, x6, x7 = x

    # ----------------------------
    # Objective Function (Weight)
    # ----------------------------
    f = (
        0.7854 * x1 * x2**2 *
        (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.4777 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2)
    )

    # ----------------------------
    # Constraints (g_i <= 0)
    # ----------------------------
    g = []

    g.append(27 / (x1 * x2**2 * x3) - 1)
    g.append(397.5 / (x1 * x2**2 * x3**2) - 1)
    g.append(1.93 * x4**3 / (x2 * x6**4 * x3) - 1)
    g.append(1.93 * x5**3 / (x2 * x7**4 * x3) - 1)

    g.append(
        ((745 * x4 / (x2 * x3))**2 + 1.69e7)**0.5 / (110 * x6**3) - 1
    )
    g.append(
        ((745 * x5 / (x2 * x3))**2 + 1.575e8)**0.5 / (85 * x7**3) - 1
    )

    g.append(x2 * x3 / 40 - 1)
    g.append(5 * x2 / x1 - 1)
    g.append(x1 / (12 * x2) - 1)
    g.append(1.5 * x6 + 1.9 / x4 - 1)
    g.append(1.1 * x7 + 1.9 / x5 - 1)

    # ----------------------------
    # Aggregate constraint violation
    # ----------------------------
    violation = sum(max(0.0, gi) for gi in g)

    return f, violation


def pressure_vessel_objective(x):
    """
    Robust pressure vessel design under manufacturing uncertainty.
    Returns (objective_value, constraint_violation)
    """
    Ts, Th, R, L = x

    # ---- Manufacturing uncertainty (noise) ----
    Ts *= 1 + random.gauss(0, 0.03)
    Th *= 1 + random.gauss(0, 0.03)
    R  *= 1 + random.gauss(0, 0.02)
    L  *= 1 + random.gauss(0, 0.01)

    # ---- Cost function (ASME model) ----
    cost = (
        0.6224 * Ts * R * L +
        1.7781 * Th * R**2 +
        3.1661 * Ts**2 * L +
        19.84  * Ts**2 * R
    )

    # ---- Constraints ----
    v = 0.0

    # Thickness constraints
    v += max(0.0, 0.0193 * R - Ts)
    v += max(0.0, 0.00954 * R - Th)

    # Volume constraint
    volume = math.pi * R**2 * L + (4.0 / 3.0) * math.pi * R**3
    v += max(0.0, 1_296_000 - volume)

    # Length constraint
    v += max(0.0, L - 240.0)

    return cost, v


# Function registry
FUNCTIONS = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Resource Allocation": resource_allocation_objective,
    "Pressure Vessel (Robust)": pressure_vessel_objective,
    "Speed Reducer": speed_reducer_objective
}


# ==============================
# Optimization Runner
# ==============================
def run_optimizer(optimizer_class, dim, pop_size, iters, bounds, obj_fn, runs):
    """
    Run optimizer multiple times and return statistics.
    """
    all_histories = []
    all_f = []
    all_v = []
    
    for _ in range(runs):
        optimizer = optimizer_class(
            dim=dim,
            pop_size=pop_size,
            iters=iters,
            bounds=bounds,
            obj_fn=obj_fn
        )
        
        optimizer.run()
        x, f, v = optimizer.best_solution()
        
        all_histories.append(optimizer.history_best)
        all_f.append(f)
        all_v.append(v)
    
    # Compute statistics
    import statistics
    mean_f = statistics.mean(all_f)
    std_f = statistics.stdev(all_f) if len(all_f) > 1 else 0
    mean_v = statistics.mean(all_v)
    
    # Mean convergence curve
    mean_history = []
    for t in range(iters):
        values_at_t = [hist[t] for hist in all_histories]
        mean_history.append(statistics.mean(values_at_t))
    
    return mean_history, mean_f, std_f, mean_v

# ==============================
# Tkinter UI
# ==============================
class OptimizationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization Algorithm Comparison")
        self.root.geometry("1000x800")
        
        # Variables
        self.function_var = tk.StringVar(value="Sphere")
        self.dim_var = tk.IntVar(value=10)
        self.pop_size_var = tk.IntVar(value=40)
        self.iters_var = tk.IntVar(value=100)
        self.runs_var = tk.IntVar(value=10)  # New: Number of runs
        
        # UI Elements
        self.create_widgets()
        
    def create_widgets(self):
        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # Function Selection
        ttk.Label(input_frame, text="Objective Function:").grid(row=0, column=0, sticky="w")
        func_combo = ttk.Combobox(input_frame, textvariable=self.function_var, 
                                  values=list(FUNCTIONS.keys()), state="readonly")
        func_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Dimension
        ttk.Label(input_frame, text="Dimension:").grid(row=1, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.dim_var).grid(row=1, column=1, padx=5, pady=2)
        
        # Population Size
        ttk.Label(input_frame, text="Population Size:").grid(row=2, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.pop_size_var).grid(row=2, column=1, padx=5, pady=2)
        
        # Iterations
        ttk.Label(input_frame, text="Iterations:").grid(row=3, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.iters_var).grid(row=3, column=1, padx=5, pady=2)
        
        # Number of Runs
        ttk.Label(input_frame, text="Number of Runs:").grid(row=4, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.runs_var).grid(row=4, column=1, padx=5, pady=2)
        
        # Run Button
        ttk.Button(input_frame, text="Run Optimization", command=self.run_optimization).grid(row=5, column=0, columnspan=2, pady=10)
        
        # Results Frame
        self.results_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Results Text
        self.results_text = tk.Text(self.results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True)
        
        # Plot Frame
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
    def run_optimization(self):
        try:
            # Get parameters
            func_name = self.function_var.get()
            dim = self.dim_var.get()
            pop_size = self.pop_size_var.get()
            iters = self.iters_var.get()
            runs = self.runs_var.get()
            
            obj_fn = FUNCTIONS[func_name]
            
            # Set bounds based on function
            if func_name == "Resource Allocation":
                bounds = [(0.0, 1.0)] * dim
                func_desc = "Resource Allocation: Distribute limited resources across units for balanced utilization."
                constraints_desc = f"Total resources = {dim * 0.1:.1f}, Each unit: [0.0, 1.0]"

            elif func_name == "Pressure Vessel (Robust)":
                # Ts, Th, R, L
                bounds = [
                    (0.1, 10.0),   # Ts
                    (0.1, 10.0),   # Th
                    (10.0, 200.0), # R
                    (10.0, 240.0)  # L
                ]
                dim = 4  # Force correct dimension
                self.dim_var.set(4)
                func_desc = "Pressure Vessel Design: Minimize cost under manufacturing uncertainty."
                constraints_desc = "Ts: [0.1, 10.0], Th: [0.1, 10.0], R: [10.0, 200.0], L: [10.0, 240.0]"

            elif func_name == "Speed Reducer":
                bounds = [
                    (2.6, 3.6),   # x1
                    (0.7, 0.8),   # x2
                    (17.0, 28.0), # x3
                    (7.3, 8.3),   # x4
                    (7.3, 8.3),   # x5
                    (2.9, 3.9),   # x6
                    (5.0, 5.6)    # x7
                ]
                dim = 7
                self.dim_var.set(7)

                func_desc = "Speed Reducer Design: Minimize weight under 11 nonlinear constraints."
                constraints_desc = "7 variables, 11 nonlinear constraints (classic engineering benchmark)"

            else:
                bounds = [(-5.12, 5.12)] * dim
                func_desc = f"{func_name}: Standard benchmark function."
                constraints_desc = f"All dimensions: [-5.12, 5.12]"

            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Display configuration
            self.results_text.insert(tk.END, f"Function: {func_desc}\n")
            self.results_text.insert(tk.END, f"Constraints: {constraints_desc}\n")
            self.results_text.insert(tk.END, f"Dimension: {dim}, Population: {pop_size}, Iterations: {iters}, Runs: {runs}\n\n")
            
            # Run optimizations
            self.results_text.insert(tk.END, f"Running optimization for {func_name} function (D={dim})...\n\n")
            
            results = {}
            
            # PSO
            pso_hist, pso_mean_f, pso_std_f, pso_mean_v = run_optimizer(PSO, dim, pop_size, iters, bounds, obj_fn, runs)
            results["PSO"] = (pso_hist, pso_mean_f, pso_std_f, pso_mean_v)
            
            # DE
            de_hist, de_mean_f, de_std_f, de_mean_v = run_optimizer(DE, dim, pop_size, iters, bounds, obj_fn, runs)
            results["DE"] = (de_hist, de_mean_f, de_std_f, de_mean_v)
            
            # GMO
            gmo_hist, gmo_mean_f, gmo_std_f, gmo_mean_v = run_optimizer(GMO, dim, pop_size, iters, bounds, obj_fn, runs)
            results["GMO"] = (gmo_hist, gmo_mean_f, gmo_std_f, gmo_mean_v)
            
            # Display results
            self.results_text.insert(tk.END, "Algorithm Results (Statistics over {} runs):\n".format(runs))
            self.results_text.insert(tk.END, "=" * 80 + "\n")
            
            for algo, (hist, mean_f, std_f, mean_v) in results.items():
                self.results_text.insert(tk.END, f"{algo}:\n")
                self.results_text.insert(tk.END, f"  Mean Best Objective: {mean_f:.6f} ± {std_f:.6f}\n")
                self.results_text.insert(tk.END, f"  Mean Constraint Violation: {mean_v:.6e}\n\n")
            
            # Create plot
            self.create_plot(results, func_name, dim)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_plot(self, results, func_name, dim):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {"PSO": "red", "DE": "green", "GMO": "blue"}
        linestyles = {"PSO": "--", "DE": ":", "GMO": "-"}
        
        for algo, (hist, mean_f, std_f, mean_v) in results.items():
            ax.plot(hist, label=f"{algo} (Mean: {mean_f:.4f} ± {std_f:.4f})", 
                   color=colors[algo], linestyle=linestyles[algo], linewidth=2)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Best Objective Value")
        ax.set_title(f"{func_name} Function Optimization (D={dim}) - Mean Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationUI(root)
    root.mainloop()