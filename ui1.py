import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ======================================================
# PLACEHOLDER OBJECTIVE FUNCTIONS
# (replace with your real ones)
# ======================================================

def speed_reducer_objective(x):
    # dummy objective for UI wiring
    return np.sum(np.square(x))

def pressure_vessel_objective(x):
    return 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]**2 + 3.1661*x[0]**2*x[3]

# ======================================================
# PLACEHOLDER GMO CLASS
# (replace with your real GMO)
# ======================================================

class GMO:
    def __init__(self, dim, pop_size, iters, bounds, obj_fn):
        self.dim = dim
        self.pop_size = pop_size
        self.iters = iters
        self.bounds = bounds
        self.obj_fn = obj_fn
        self.history_best = []

    def run(self):
        best = float("inf")
        for i in range(self.iters):
            x = np.array([
                np.random.uniform(b[0], b[1]) for b in self.bounds
            ])
            f = self.obj_fn(x)
            best = min(best, f)
            self.history_best.append(best)

        self.best_x = x
        self.best_f = best
        self.best_v = 0.0  # assume feasible for UI demo

    def best_solution(self):
        return self.best_x, self.best_f, self.best_v


# ======================================================
# SEPARATE DESIGN PAGE
# ======================================================

class GMODesignPage:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Engineering Design using GMO")
        self.root.geometry("1000x700")

        self.problem_var = tk.StringVar(value="Speed Reducer")
        self.pop_var = tk.IntVar(value=40)
        self.iter_var = tk.IntVar(value=100)

        self.build_ui()
        self.root.mainloop()

    def build_ui(self):
        # ------------------------------
        # Header
        # ------------------------------
        header = ttk.Label(
            self.root,
            text="GMO-based Engineering Design Studio",
            font=("Segoe UI", 16, "bold")
        )
        header.pack(pady=10)

        # ------------------------------
        # Control Panel
        # ------------------------------
        control = ttk.LabelFrame(self.root, text="Design Configuration", padding=10)
        control.pack(fill="x", padx=15)

        ttk.Label(control, text="Problem:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            control,
            textvariable=self.problem_var,
            values=["Speed Reducer", "Pressure Vessel"],
            state="readonly",
            width=18
        ).grid(row=0, column=1, padx=5)

        ttk.Label(control, text="Population:").grid(row=0, column=2)
        ttk.Entry(control, textvariable=self.pop_var, width=8).grid(row=0, column=3)

        ttk.Label(control, text="Iterations:").grid(row=0, column=4)
        ttk.Entry(control, textvariable=self.iter_var, width=8).grid(row=0, column=5)

        ttk.Button(
            control,
            text="Run GMO",
            command=self.run_gmo
        ).grid(row=0, column=6, padx=10)

        # ------------------------------
        # Output Area
        # ------------------------------
        self.output = tk.Text(self.root, height=10, font=("Consolas", 10))
        self.output.pack(fill="x", padx=15, pady=10)

        # ------------------------------
        # Plot Area
        # ------------------------------
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True, padx=15, pady=5)

    def run_gmo(self):
        self.output.delete(1.0, tk.END)
        for w in self.plot_frame.winfo_children():
            w.destroy()

        problem = self.problem_var.get()

        if problem == "Speed Reducer":
            dim = 7
            bounds = [
                (2.6, 3.6), (0.7, 0.8), (17, 28),
                (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5.0, 5.6)
            ]
            obj = speed_reducer_objective
            names = ["b", "m", "z", "l1", "l2", "d1", "d2"]

        else:
            dim = 4
            bounds = [(0.1, 10), (0.1, 10), (10, 200), (10, 240)]
            obj = pressure_vessel_objective
            names = ["Ts", "Th", "R", "L"]

        gmo = GMO(
            dim=dim,
            pop_size=self.pop_var.get(),
            iters=self.iter_var.get(),
            bounds=bounds,
            obj_fn=obj
        )

        gmo.run()
        x, f, v = gmo.best_solution()

        # ------------------------------
        # Display Results
        # ------------------------------
        self.output.insert(tk.END, f"Problem: {problem}\n")
        self.output.insert(tk.END, f"Best Objective Value: {f:.4f}\n")
        self.output.insert(tk.END, f"Constraint Violation: {v:.6e}\n\n")
        self.output.insert(tk.END, "Optimized Design Variables:\n")

        for n, val in zip(names, x):
            self.output.insert(tk.END, f"  {n} = {val:.4f}\n")

        self.output.insert(
            tk.END,
            "\n✔ FEASIBLE DESIGN\n" if v == 0 else "\n✘ INFEASIBLE DESIGN\n"
        )

        # ------------------------------
        # Convergence Plot
        # ------------------------------
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(gmo.history_best, linewidth=2)
        ax.set_title("GMO Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Objective")
        ax.grid(alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ======================================================
# RUN AS STANDALONE PAGE
# ======================================================
if __name__ == "__main__":
    GMODesignPage()
