import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import time
import google.generativeai as genai

from algorithms.gmo import GMO

# ======================================================
# CONFIG
# ======================================================
CPU_COST = 30
RAM_COST = 5
REPLICA_COST = 20
BUDGET = 1200

SLA_TARGET = 0.99
Z_SLA = 2.33
MIN_REPLICAS = 2
GEMINI_API_KEY="AIzaSyDsVkEwjYreUJ1gMhIdeJJGSQ05GQG9BTE"


# ======================================================
# DEPLOYABLE CLOUD OBJECTIVE
# ======================================================
def deployable_cloud_objective(x, traffic_level, n_services):
    base_demand = {
        "Normal": 1.0,
        "Peak": 1.6,
        "Flash Sale": 2.4
    }[traffic_level]

    infra_cost = 0.0
    sla_violation = 0.0
    budget_violation = 0.0
    cpu_alloc = []

    for i in range(n_services):
        cpu = float(x[3*i])
        ram = float(x[3*i + 1])
        rep = int(round(x[3*i + 2]))

        if rep < MIN_REPLICAS:
            sla_violation += (MIN_REPLICAS - rep) ** 2
            rep = MIN_REPLICAS

        mean = base_demand
        std = 0.15 * base_demand
        required = mean + Z_SLA * std
        available = cpu * rep

        if available < required:
            sla_violation += (required - available) ** 2

        infra_cost += (
            cpu * CPU_COST +
            ram * RAM_COST +
            rep * REPLICA_COST
        )

        cpu_alloc.append(cpu)

    arch_imbalance = np.std(cpu_alloc)

    if infra_cost > BUDGET:
        budget_violation = (infra_cost - BUDGET) ** 2

    total_violation = sla_violation + budget_violation
    objective = infra_cost + 40 * arch_imbalance

    diagnostics = {
        "infra_cost": infra_cost,
        "sla_penalty": sla_violation,
        "arch_penalty": arch_imbalance,
        "budget_penalty": budget_violation
    }

    return objective, total_violation, diagnostics


# ======================================================
# GEMINI INSIGHTS
# ======================================================
def get_gemini_insights(prompt):
    try:
        print(GEMINI_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        model_name: str = "gemma-3-27b-it"
        model = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(
            temperature=0,
            top_p=1,
            top_k=1,
        )
    )
        response = model.generate_content(prompt)
        print(response)
        return response.text
    except Exception as e:
        print(e)
        return "⚠ AI insights unavailable. Check API key or network."


def explain_metrics(diag):
    lines = []

    if diag["sla_penalty"] == 0:
        lines.append("✔ SLA targets are fully satisfied.")
    else:
        lines.append("⚠ Potential SLA risks detected under peak load.")

    if diag["arch_penalty"] < 0.4:
        lines.append("✔ CPU allocation is well balanced across services.")
    else:
        lines.append("⚠ CPU imbalance detected — consider consolidation.")

    if diag["budget_penalty"] == 0:
        lines.append("✔ Deployment is within budget.")
    else:
        lines.append("⚠ Budget constraints violated.")

    return "\n".join(lines)


# ======================================================
# UI
# ======================================================
class CloudResourceUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deployable Cloud Architecture Optimizer")
        self.root.geometry("1200x850")

        self.services_var = tk.IntVar(value=6)
        self.traffic_var = tk.StringVar(value="Normal")

        self.create_ui()

    def create_ui(self):
        cfg = ttk.LabelFrame(self.root, text="Scenario Configuration", padding=10)
        cfg.pack(fill="x", padx=10, pady=5)

        ttk.Label(cfg, text="Microservices").grid(row=0, column=0)
        ttk.Entry(cfg, textvariable=self.services_var, width=10).grid(row=0, column=1)

        ttk.Label(cfg, text="Traffic Profile").grid(row=1, column=0)
        ttk.Combobox(
            cfg, textvariable=self.traffic_var,
            values=["Normal", "Peak", "Flash Sale"], state="readonly"
        ).grid(row=1, column=1)

        ttk.Button(cfg, text="Run Optimization", command=self.run_optimization)\
            .grid(row=2, column=0, columnspan=2, pady=8)

        self.progress = ttk.Progressbar(self.root, length=400, mode="determinate")
        self.progress.pack(pady=5)

        self.output = ttk.LabelFrame(self.root, text="Results", padding=10)
        self.output.pack(fill="both", expand=True, padx=10, pady=5)

        self.text = tk.Text(self.output, height=12)
        self.text.pack(fill="x")

        self.insights = ttk.LabelFrame(self.root, text="AI Architecture Insights", padding=10)
        self.insights.pack(fill="x", padx=10, pady=5)

        self.insight_text = tk.Text(self.insights, height=6, wrap="word")
        self.insight_text.pack(fill="x")

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True)

    def run_optimization(self):
        try:
            n = self.services_var.get()
            traffic = self.traffic_var.get()

            dim = 3 * n
            bounds = []
            for _ in range(n):
                bounds.extend([(0.2, 4.0), (0.5, 16.0), (1, 10)])

            def obj(x):
                return deployable_cloud_objective(x, traffic, n)[:2]

            gmo = GMO(dim=dim, pop_size=50, iters=120, bounds=bounds, obj_fn=obj)

            self.progress["value"] = 0
            self.root.update()

            for i in range(gmo.max_iters):
                gmo.step()
                self.progress["value"] = (i / gmo.max_iters) * 100
                self.root.update()

            x, f, v = gmo.best_solution()
            _, _, diag = deployable_cloud_objective(x, traffic, n)

            self.show_results(x, f, v, diag, traffic, n)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_results(self, x, f, v, diag, traffic, n):
        self.text.delete(1.0, tk.END)
        self.insight_text.delete(1.0, tk.END)
        for w in self.plot_frame.winfo_children():
            w.destroy()

        self.text.insert(tk.END, "✔ Optimization Completed Successfully\n\n")
        self.text.insert(tk.END, f"Infrastructure Cost: ${diag['infra_cost']:.2f}\n")
        self.text.insert(tk.END, f"SLA Penalty: {diag['sla_penalty']:.4f}\n")
        self.text.insert(tk.END, f"Architecture Imbalance: {diag['arch_penalty']:.4f}\n")
        self.text.insert(tk.END, f"Budget Penalty: ${diag['budget_penalty']:.2f}\n\n")
        self.text.insert(tk.END, f"Final Optimization Score: {f:.2f}\n")
        self.text.insert(tk.END, f"Total Constraint Violation: {v:.2f}\n\n")

        cpu, ram, rep = [], [], []

        for i in range(n):
            cpu.append(x[3*i])
            ram.append(x[3*i + 1])
            rep.append(int(round(x[3*i + 2])))

            self.text.insert(
                tk.END,
                f"Service {i+1}: CPU={cpu[-1]:.2f} | RAM={ram[-1]:.1f}GB | Replicas={rep[-1]}\n"
            )

        explanation = explain_metrics(diag)

        prompt = f"""
Traffic Profile: {traffic}
{explanation}

CPU: {cpu}
RAM: {ram}
Replicas: {rep}

Explain this architecture simply and suggest one improvement.
"""

        self.insight_text.insert(tk.END, get_gemini_insights(prompt))

        self.animate_charts(cpu, ram, rep)

    def animate_charts(self, cpu, ram, rep):
        fig, ax = plt.subplots(1, 3, figsize=(11, 3))
        data = [cpu, ram, rep]
        titles = ["CPU Allocation", "RAM Allocation", "Replica Count"]

        bars = []
        for i in range(3):
            bars.append(ax[i].bar(range(1, len(cpu)+1), [0]*len(cpu)))
            ax[i].set_title(titles[i])
            ax[i].set_xlabel("Service")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        for step in range(20):
            for i in range(3):
                for b, v in zip(bars[i], data[i]):
                    b.set_height(min(v, b.get_height() + v / 20))
            canvas.draw()
            self.root.update()
            time.sleep(0.03)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CloudResourceUI(root)
    root.mainloop()
