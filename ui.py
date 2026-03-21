import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.gmo import GMO

# ======================================================
# CONFIG
# ======================================================

ASSETS = ["Stocks", "Bonds", "Crypto", "Gold"]
RETURNS = [0.10, 0.05, 0.25, 0.07]   # expected annual returns
VOLATILITY = [0.15, 0.05, 0.40, 0.10]


# ======================================================
# OBJECTIVE FUNCTION
# ======================================================

def portfolio_objective(x):
    total = sum(x)

    # Normalize allocations
    weights = [xi / (total + 1e-10) for xi in x]

    # Geometric growth approximation (log form)
    log_growth = 0.0
    for i in range(len(weights)):
        log_growth += weights[i] * np.log(1 + RETURNS[i])

    growth = np.exp(log_growth)

    # Risk penalty
    risk = sum(weights[i] * VOLATILITY[i] for i in range(len(weights)))

    # Constraint: sum must be 1
    violation = (total - 1.0) ** 2

    # Final objective (minimize negative growth + risk)
    objective = -growth + 0.5 * risk

    diagnostics = {
        "growth": growth,
        "risk": risk,
        "weights": weights
    }

    return objective, violation, diagnostics


def explain_metrics(diag):
    lines = []

    if diag["growth"] > 1.1:
        lines.append("✔ Strong expected portfolio growth.")
    else:
        lines.append("⚠ Low growth potential.")

    if diag["risk"] < 0.15:
        lines.append("✔ Portfolio risk is well controlled.")
    else:
        lines.append("⚠ High volatility exposure.")

    return "\n".join(lines)


# ======================================================
# UI
# ======================================================

class PortfolioUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Portfolio Optimizer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f4f8")

        self.asset_count = tk.IntVar(value=4)

        self.create_ui()

    def create_ui(self):
        header = ttk.Label(self.root, text="📈 Portfolio Optimizer", font=("Segoe UI", 18, "bold"))
        header.pack(pady=10)

        btn = ttk.Button(self.root, text="▶ Run Optimization", command=self.run_optimization)
        btn.pack(pady=10)

        self.text = tk.Text(self.root, height=10)
        self.text.pack(fill="x", padx=20)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True)

    def run_optimization(self):
        try:
            dim = len(ASSETS)
            bounds = [(0.0, 1.0)] * dim

            def obj(x):
                return portfolio_objective(x)[:2]

            gmo = GMO(dim=dim, pop_size=40, iters=500, bounds=bounds, obj_fn=obj)

            for _ in range(gmo.max_iters):
                gmo.step()

            x, f, v = gmo.best_solution()
            _, _, diag = portfolio_objective(x)

            self.show_results(x, f, diag)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_results(self, x, f, diag):
        self.text.delete(1.0, tk.END)

        weights = diag["weights"]

        self.text.insert(tk.END, "✓ OPTIMIZATION COMPLETE\n")
        self.text.insert(tk.END, "-"*40 + "\n")

        self.text.insert(tk.END, f"📈 Growth: {diag['growth']:.4f}\n")
        self.text.insert(tk.END, f"⚠ Risk: {diag['risk']:.4f}\n\n")

        for i, asset in enumerate(ASSETS):
            self.text.insert(tk.END, f"{asset}: {weights[i]*100:.2f}%\n")

        self.draw_chart(weights)

    def draw_chart(self, weights):
        for w in self.plot_frame.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(weights, labels=ASSETS, autopct='%1.1f%%')
        ax.set_title("Portfolio Allocation")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioUI(root)
    root.mainloop()