import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random

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

# Function registry
FUNCTIONS = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Resource Allocation": resource_allocation_objective
}

# ==============================
# Optimization Runner
# ==============================
def run_optimizer(optimizer_class, dim, pop_size, iters, bounds, obj_fn):
    optimizer = optimizer_class(
        dim=dim,
        pop_size=pop_size,
        iters=iters,
        bounds=bounds,
        obj_fn=obj_fn
    )
    
    optimizer.run()
    x, f, v = optimizer.best_solution()
    
    return optimizer.history_best, x, f, v

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
        
        # Run Button
        ttk.Button(input_frame, text="Run Optimization", command=self.run_optimization).grid(row=4, column=0, columnspan=2, pady=10)
        
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
            
            obj_fn = FUNCTIONS[func_name]
            
            # Set bounds based on function
            if func_name == "Resource Allocation":
                bounds = [(0.0, 1.0)] * dim
            else:
                bounds = [(-5.12, 5.12)] * dim
            
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Run optimizations
            self.results_text.insert(tk.END, f"Running optimization for {func_name} function (D={dim})...\n\n")
            
            results = {}
            
            # PSO
            pso_hist, pso_x, pso_f, pso_v = run_optimizer(PSO, dim, pop_size, iters, bounds, obj_fn)
            results["PSO"] = (pso_hist, pso_x, pso_f, pso_v)
            
            # DE
            de_hist, de_x, de_f, de_v = run_optimizer(DE, dim, pop_size, iters, bounds, obj_fn)
            results["DE"] = (de_hist, de_x, de_f, de_v)
            
            # GMO
            gmo_hist, gmo_x, gmo_f, gmo_v = run_optimizer(GMO, dim, pop_size, iters, bounds, obj_fn)
            results["GMO"] = (gmo_hist, gmo_x, gmo_f, gmo_v)
            
            # Display results
            self.results_text.insert(tk.END, "Algorithm | Best Objective | Constraint Violation\n")
            self.results_text.insert(tk.END, "-" * 50 + "\n")
            
            for algo, (hist, x, f, v) in results.items():
                self.results_text.insert(tk.END, f"{algo:<9} | {f:>14.6f} | {v:>19.6e}\n")
            
            # Create plot
            self.create_plot(results, func_name, dim)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_plot(self, results, func_name, dim):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {"PSO": "red", "DE": "green", "GMO": "blue"}
        linestyles = {"PSO": "--", "DE": ":", "GMO": "-"}
        
        for algo, (hist, x, f, v) in results.items():
            ax.plot(hist, label=f"{algo} (Best: {f:.4f})", 
                   color=colors[algo], linestyle=linestyles[algo], linewidth=2)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Objective Value")
        ax.set_title(f"{func_name} Function Optimization (D={dim})")
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