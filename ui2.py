import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

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


# ======================================================
# DEPLOYABLE CLOUD OBJECTIVE
# ======================================================


def deployable_cloud_objective_with_ha(x, traffic_level, n_services):
    # Constants for Reliability (Define these in your config)
    HA_THRESHOLD = 2  # Minimum replicas for high availability
    HA_PENALTY_WEIGHT = 500.0  # High weight to discourage SPOF

    base_demand = {
        "Normal": 1.0,
        "Peak": 1.6,
        "Flash Sale": 2.4
    }[traffic_level]

    infra_cost = 0.0
    sla_violation = 0.0
    budget_violation = 0.0
    ha_violation = 0.0  # <--- New Reliability Metric
    cpu_alloc = []

    for i in range(n_services):
        cpu = float(x[3*i])
        ram = float(x[3*i + 1])
        rep = int(round(x[3*i + 2]))

        # 1. Physical Constraint (Min Replicas for logic)
        if rep < MIN_REPLICAS:
            sla_violation += (MIN_REPLICAS - rep) ** 2
            rep = MIN_REPLICAS

        # 2. Reliability Constraint (HA Penalty)
        # If replicas are less than 2, apply a penalty for SPOF
        if rep < HA_THRESHOLD:
            ha_violation += (HA_THRESHOLD - rep) * HA_PENALTY_WEIGHT

        # 3. Demand Satisfaction
        mean = base_demand
        std = 0.15 * base_demand
        required = mean + Z_SLA * std
        available = cpu * rep

        if available < required:
            sla_violation += (required - available) ** 2

        # 4. Cost Calculation
        infra_cost += (
            cpu * CPU_COST +
            ram * RAM_COST +
            rep * REPLICA_COST
        )

        cpu_alloc.append(cpu)

    arch_imbalance = np.std(cpu_alloc)

    # 5. Budget Constraint
    if infra_cost > BUDGET:
        budget_violation = (infra_cost - BUDGET) ** 2

    # Total Violation includes Reliability now
    total_violation = sla_violation + budget_violation + ha_violation
    
    # Final Objective (The value GMO tries to minimize)
    objective = infra_cost + (40 * arch_imbalance) + ha_violation

    diagnostics = {
        "infra_cost": infra_cost,
        "sla_penalty": sla_violation,
        "arch_penalty": arch_imbalance,
        "budget_penalty": budget_violation,
        "ha_penalty": ha_violation  # <--- Track this in your Radar/Pie charts
    }

    return objective, total_violation, diagnostics



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
        self.root.title("☁ Cloud Architecture Optimizer")
        self.root.geometry("1400x1000")
        self.root.state('zoomed')  # Maximize window
        self.root.configure(bg="#f0f4f8")

        # Apply a modern theme
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Custom styles
        self.style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#2c3e50")
        self.style.configure("TLabelframe", background="#f0f4f8")
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground="#34495e")
        self.style.configure("TLabel", font=("Segoe UI", 10), background="#f0f4f8")
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        self.style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        self.style.configure("TEntry", padding=5)
        self.style.configure("TCombobox", padding=5)
        
        self.services_var = tk.IntVar(value=6)
        self.traffic_var = tk.StringVar(value="Normal")

        self.create_ui()

    def create_ui(self):
        # Header
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=20, pady=(15, 5))
        ttk.Label(header, text="☁ Cloud Architecture Optimizer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Optimize microservice resources using AI-powered algorithms", 
                  font=("Segoe UI", 9), foreground="#7f8c8d").pack(anchor="w")

        # Configuration Panel
        cfg = ttk.LabelFrame(self.root, text="⚙ Configuration", padding=15)
        cfg.pack(fill="x", padx=20, pady=10)

        # Left side - inputs
        input_frame = ttk.Frame(cfg)
        input_frame.pack(side="left", padx=10)

        ttk.Label(input_frame, text="Number of Microservices:").grid(row=0, column=0, sticky="w", pady=5)
        svc_entry = ttk.Entry(input_frame, textvariable=self.services_var, width=12, font=("Segoe UI", 10))
        svc_entry.grid(row=0, column=1, padx=(10, 0), pady=5)

        ttk.Label(input_frame, text="Traffic Profile:").grid(row=1, column=0, sticky="w", pady=5)
        traffic_combo = ttk.Combobox(
            input_frame, textvariable=self.traffic_var,
            values=["Normal", "Peak", "Flash Sale"], state="readonly", width=14, font=("Segoe UI", 10)
        )
        traffic_combo.grid(row=1, column=1, padx=(10, 0), pady=5)

        # Right side - button
        btn_frame = ttk.Frame(cfg)
        btn_frame.pack(side="right", padx=20)
        
        self.run_btn = ttk.Button(btn_frame, text="▶ Run Optimization", 
                                   command=self.run_optimization, style="Accent.TButton")
        self.run_btn.pack(pady=10)

        # Status bar with progress
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=20, pady=3)
        
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="#27ae60", 
                                       font=("Segoe UI", 9, "italic"))
        self.status_label.pack(side="left")
        
        self.progress = ttk.Progressbar(status_frame, length=300, mode="determinate")
        self.progress.pack(side="right")

        # Middle section: Results side by side with charts - COMPACT
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill="x", padx=20, pady=3)

        # Results Panel (left) - smaller
        self.output = ttk.LabelFrame(middle_frame, text="📊 Results", padding=8)
        self.output.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Combined scrollable text area for results
        self.text = tk.Text(self.output, height=8, font=("Consolas", 8), 
                            bg="#fafafa", fg="#2c3e50", relief="flat", padx=6, pady=6)
        self.text.pack(fill="both", expand=True)
        self.text.insert(tk.END, "Click 'Run Optimization' to start...")
        self.text.config(state="disabled")

        # Charts Panel - LARGER (gets more space now)
        self.plot_frame = ttk.LabelFrame(self.root, text="📈 Optimization Charts", padding=10)
        self.plot_frame.pack(fill="both", expand=True, padx=20, pady=(3, 10))

    def run_optimization(self):
        try:
            n = self.services_var.get()
            traffic = self.traffic_var.get()

            # Update status
            self.status_label.config(text="⏳ Optimizing...", foreground="#e67e22")
            self.run_btn.config(state="disabled")
            self.root.update()

            dim = 3 * n
            bounds = []
            for _ in range(n):
                bounds.extend([(0.2, 4.0), (0.5, 16.0), (1, 10)])

            def obj(x):
                return deployable_cloud_objective_with_ha(x, traffic, n)[:2]

            gmo = GMO(dim=dim, pop_size=50, iters=120, bounds=bounds, obj_fn=obj)

            self.progress["value"] = 0
            self.root.update()

            # Track iteration history
            self.iteration_history = []
            
            for i in range(gmo.max_iters):
                gmo.step()
                self.progress["value"] = ((i + 1) / gmo.max_iters) * 100
                
                # Get current best solution and track it
                x_curr, f_curr, v_curr = gmo.best_solution()
                cpu_iter = [x_curr[3*j] for j in range(n)]
                ram_iter = [x_curr[3*j + 1] for j in range(n)]
                rep_iter = [int(round(x_curr[3*j + 2])) for j in range(n)]
                
                self.iteration_history.append({
                    'iteration': i + 1,
                    'cpu': cpu_iter[:],
                    'ram': ram_iter[:],
                    'replicas': rep_iter[:],
                    'cost': f_curr,
                    'violation': v_curr
                })
                
                self.root.update()

            x, f, v = gmo.best_solution()
            _, _, diag = deployable_cloud_objective_with_ha(x, traffic, n)

            self.status_label.config(text="✓ Complete", foreground="#27ae60")
            self.run_btn.config(state="normal")
            self.show_results(x, f, v, diag, traffic, n)

        except Exception as e:
            self.status_label.config(text="✗ Error", foreground="#e74c3c")
            self.run_btn.config(state="normal")
            messagebox.showerror("Error", str(e))

    def show_results(self, x, f, v, diag, traffic, n):
        # Enable text widget for editing
        self.text.config(state="normal")
        
        self.text.delete(1.0, tk.END)
        for w in self.plot_frame.winfo_children():
            w.destroy()

        # Results with compact formatting
        self.text.insert(tk.END, "✓ OPTIMIZATION COMPLETE\n")
        self.text.insert(tk.END, "─" * 40 + "\n")
        self.text.insert(tk.END, f"💰 Cost: ${diag['infra_cost']:.2f}  |  🎯 Score: {f:.2f}\n")
        self.text.insert(tk.END, f"⚠ SLA: {diag['sla_penalty']:.4f}  |  ⚖ Imbalance: {diag['arch_penalty']:.4f}\n")
        self.text.insert(tk.END, "─" * 40 + "\n")

        cpu, ram, rep = [], [], []

        for i in range(n):
            cpu.append(x[3*i])
            ram.append(x[3*i + 1])
            rep.append(int(round(x[3*i + 2])))
            self.text.insert(tk.END, f"S{i+1}: CPU={cpu[-1]:.2f} RAM={ram[-1]:.1f}GB Rep={rep[-1]}\n")

        explanation = explain_metrics(diag)
        
        self.text.config(state="disabled")

        self.show_charts(cpu, ram, rep, n)

    def show_charts(self, cpu, ram, rep, n):
        # Clear previous charts
        for w in self.plot_frame.winfo_children():
            w.destroy()
        
        # Create figure with Resource Summary chart only
        fig = plt.figure(figsize=(14, 4))
        fig.patch.set_facecolor('#f0f4f8')
        
        ax = fig.add_subplot(111)
        
        # Draw combined resource summary chart
        self.draw_resource_summary(ax, cpu, ram, rep, n)
        
        plt.tight_layout(pad=2.0)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add iteration details button
        self.add_iteration_details_panel(n)

    def draw_resource_summary(self, ax, cpu, ram, rep, n):
        """Draw a grouped bar chart showing all resources per service"""
        ax.set_facecolor('#fafafa')
        
        services = list(range(1, n + 1))
        x = np.arange(n)
        width = 0.25
        
        # Normalize data for comparison (scale RAM down, keep CPU and replicas similar)
        cpu_norm = cpu
        ram_norm = [r / 4 for r in ram]  # Scale RAM for visual comparison
        rep_norm = rep
        
        bars1 = ax.bar(x - width, cpu_norm, width, label='CPU (cores)', color='#3498db', edgecolor='white')
        bars2 = ax.bar(x, ram_norm, width, label='RAM (GB/4)', color='#2ecc71', edgecolor='white')
        bars3 = ax.bar(x + width, rep_norm, width, label='Replicas', color='#9b59b6', edgecolor='white')
        
        ax.set_title("Resource Summary by Service", fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
        ax.set_xlabel("Service", fontsize=10, color='#7f8c8d')
        ax.set_ylabel("Value", fontsize=10, color='#7f8c8d')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{i}' for i in services], fontsize=10)
        ax.tick_params(colors='#7f8c8d', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    def add_iteration_details_panel(self, n):
        """Add a button to view detailed iteration history"""
        btn_frame = ttk.Frame(self.plot_frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="📋 View Iteration Details", 
                   command=lambda: self.show_iteration_popup(n)).pack(side="left", padx=10)
        
        # Show summary of key iterations
        if hasattr(self, 'iteration_history') and self.iteration_history:
            summary_label = ttk.Label(btn_frame, 
                text=f"Total iterations: {len(self.iteration_history)} | "
                     f"Initial cost: {self.iteration_history[0]['cost']:.2f} | "
                     f"Final cost: {self.iteration_history[-1]['cost']:.2f}",
                font=("Segoe UI", 9), foreground="#7f8c8d")
            summary_label.pack(side="right", padx=10)

    def show_iteration_popup(self, n):
        """Show popup window with iteration-by-iteration resource allocations"""
        popup = tk.Toplevel(self.root)
        popup.title("📊 Iteration Details - Resource Allocations")
        popup.geometry("900x600")
        popup.configure(bg="#f0f4f8")
        
        # Header
        header = ttk.Label(popup, text="Resource Allocation History", 
                          font=("Segoe UI", 14, "bold"))
        header.pack(pady=10)
        
        # Create scrollable frame
        container = ttk.Frame(popup)
        container.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(container, bg="#fafafa")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Table headers
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        headers = ["Iter", "Cost", "Violation"]
        for svc in range(n):
            headers.extend([f"S{svc+1} CPU", f"S{svc+1} RAM"])
        
        for col, h in enumerate(headers):
            lbl = ttk.Label(header_frame, text=h, font=("Consolas", 9, "bold"), 
                           width=8, anchor="center")
            lbl.grid(row=0, column=col, padx=2, pady=2)
        
        # Show every 10th iteration + first and last
        if hasattr(self, 'iteration_history') and self.iteration_history:
            display_iters = []
            for i, hist in enumerate(self.iteration_history):
                if i == 0 or i == len(self.iteration_history) - 1 or (i + 1) % 10 == 0:
                    display_iters.append(hist)
            
            for row_idx, hist in enumerate(display_iters):
                row_frame = ttk.Frame(scrollable_frame)
                row_frame.pack(fill="x", padx=5, pady=1)
                
                # Alternate row colors
                bg_color = "#ffffff" if row_idx % 2 == 0 else "#f5f5f5"
                
                values = [
                    f"{hist['iteration']}",
                    f"{hist['cost']:.1f}",
                    f"{hist['violation']:.2f}"
                ]
                
                for svc in range(n):
                    values.append(f"{hist['cpu'][svc]:.2f}")
                    values.append(f"{hist['ram'][svc]:.1f}")
                
                for col, val in enumerate(values):
                    lbl = tk.Label(row_frame, text=val, font=("Consolas", 9), 
                                  width=8, anchor="center", bg=bg_color)
                    lbl.grid(row=0, column=col, padx=2, pady=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CloudResourceUI(root)
    root.mainloop()
