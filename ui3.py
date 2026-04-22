import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math

# Keep your original import
from algorithms.gmo import GMO


# =========================
# TAB 1 OBJECTIVE FUNCTION
# =========================
def simple_parabola(x):
    # f(x) = x^2, with 0 constraint violations
    return x[0]**2, 0


# =========================
# UI DASHBOARD
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("GMO Optimization Dashboard")
        self.root.geometry("950x750")

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
        # Input Frame
        input_frame = tk.Frame(self.tab2)
        input_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)

        # 1. Objective Function
        tk.Label(input_frame, text="Objective Function f(x):").grid(row=0, column=0, sticky="w", pady=5)
        self.func_entry = tk.Entry(input_frame, width=50)
        self.func_entry.insert(0, "x[0]**2 + x[1]**2")
        self.func_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(input_frame, text="(Use python syntax, e.g., x[0]**2 + math.sin(x[1]))").grid(row=0, column=2, sticky="w")

        # 2. Dimensions
        tk.Label(input_frame, text="Dimensions:").grid(row=1, column=0, sticky="w", pady=5)
        self.dim_entry = tk.Entry(input_frame, width=15)
        self.dim_entry.insert(0, "2")
        self.dim_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # 3. Bounds
        tk.Label(input_frame, text="Bounds (min,max;...):").grid(row=2, column=0, sticky="w", pady=5)
        self.bounds_entry = tk.Entry(input_frame, width=50)
        self.bounds_entry.insert(0, "-10,10;-10,10")
        self.bounds_entry.grid(row=2, column=1, padx=5, pady=5)

        # 4. Population Size
        tk.Label(input_frame, text="Population Size:").grid(row=3, column=0, sticky="w", pady=5)
        self.pop_entry = tk.Entry(input_frame, width=15)
        self.pop_entry.insert(0, "30")
        self.pop_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # 5. Iterations
        tk.Label(input_frame, text="Iterations:").grid(row=4, column=0, sticky="w", pady=5)
        self.iter_entry = tk.Entry(input_frame, width=15)
        self.iter_entry.insert(0, "100")
        self.iter_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Run Button
        tk.Button(input_frame, text="Run Optimization", command=self.run_custom, font=("Arial", 10, "bold"), bg="#ddd").grid(row=5, column=0, columnspan=2, pady=15)

        # Output Text
        out_frame = tk.Frame(self.tab2)
        out_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)

        self.output2 = tk.Text(out_frame, height=8, font=("Courier", 10))
        self.output2.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Plot Frame
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=out_frame)
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def run_custom(self):
        try:
            # Parse inputs
            func_str = self.func_entry.get()
            dim = int(self.dim_entry.get())
            pop_size = int(self.pop_entry.get())
            iters = int(self.iter_entry.get())
            
            # Parse bounds
            bounds_str = self.bounds_entry.get().split(';')
            bounds = []
            for b in bounds_str:
                parts = b.split(',')
                bounds.append((float(parts[0]), float(parts[1])))
            
            if len(bounds) != dim:
                messagebox.showerror("Configuration Error", f"You specified {dim} dimensions but provided {len(bounds)} bounds.\nEnsure bounds are separated by a semicolon (;).")
                return

            # Safely build the custom objective function string for eval()
            def custom_obj(x):
                # We inject the python math module and the agent's x-array
                allowed_locals = {"x": x, "math": math}
                val = eval(func_str, {"__builtins__": None}, allowed_locals)
                return val, 0  # Assuming 0 constraints for the custom quick-tester

            # Execute GMO
            gmo = GMO(dim=dim, pop_size=pop_size, iters=iters, bounds=bounds, obj_fn=custom_obj)
            
            history = []
            for _ in range(iters):
                gmo.step()
                _, f, _ = gmo.best_solution()
                history.append(f)

            best_x, best_f, best_v = gmo.best_solution()

            # Write Output
            self.output2.delete(1.0, tk.END)
            self.output2.insert(tk.END, "--- OPTIMIZATION COMPLETE ---\n")
            self.output2.insert(tk.END, f"Objective Value: {best_f:.8f}\n\n")
            self.output2.insert(tk.END, "Best Variables Found:\n")
            for i, val in enumerate(best_x):
                self.output2.insert(tk.END, f"  x[{i}] = {val:.6f}\n")

            # Draw Convergence Plot
            self.ax2.clear()
            self.ax2.plot(history, color="green", linewidth=2)
            self.ax2.set_title("Convergence Curve")
            self.ax2.set_xlabel("Iteration")
            self.ax2.set_ylabel("Best Cost")
            self.ax2.grid(True, linestyle="--", alpha=0.6)
            self.canvas2.draw()

        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to run optimization.\n\nDetails: {str(e)}")


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()