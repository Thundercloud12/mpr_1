import tkinter as tk
import matplotlib.pyplot as plt

from algorithms.gmo import GMO


# =========================
# PRESSURE VESSEL FUNCTION
# =========================
def pressure_vessel(x):
    x1, x2, x3, x4 = x

    # Objective
    f = (
        0.6224 * x1 * x3 * x4 +
        1.7781 * x2 * x3 * x3 +
        3.1661 * x1 * x1 * x4 +
        19.84 * x1 * x1 * x3
    )

    # Constraints (<= 0)
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = -3.14159 * x3 * x3 * x4 - (4/3) * 3.14159 * x3**3 + 1296000
    g4 = x4 - 240

    violation = 0
    for g in [g1, g2, g3, g4]:
        if g > 0:
            violation += g

    return f, violation


# =========================
# UI
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("GMO - Pressure Vessel Optimization")

        tk.Button(root, text="Run Optimization", command=self.run).pack(pady=10)

        self.output = tk.Text(root, height=15, width=60)
        self.output.pack()

    def run(self):
        dim = 4
        bounds = [
            (0.1, 100),  # x1
            (0.1, 100),  # x2
            (10, 200),   # x3
            (10, 200)    # x4
        ]

        gmo = GMO(dim=dim, pop_size=30, iters=300, bounds=bounds, obj_fn=pressure_vessel)

        history = []

        for _ in range(300):
            gmo.step()
            _, f, v = gmo.best_solution()
            history.append(f)

        x, f, v = gmo.best_solution()

        # Display result
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "Best Solution:\n")
        self.output.insert(tk.END, f"x1={x[0]:.4f}\n")
        self.output.insert(tk.END, f"x2={x[1]:.4f}\n")
        self.output.insert(tk.END, f"x3={x[2]:.4f}\n")
        self.output.insert(tk.END, f"x4={x[3]:.4f}\n\n")

        self.output.insert(tk.END, f"Cost: {f:.4f}\n")
        self.output.insert(tk.END, f"Violation: {v:.6f}\n")

        # Plot convergence
        plt.plot(history)
        plt.title("Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()


# =========================
# MAIN
# =========================
root = tk.Tk()
app = App(root)
root.mainloop()