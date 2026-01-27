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
GEMINI_API_KEY="AIzaSyDuOnBdwzYEvzWlyL0IeW03PSxUhK6l-1M"


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

        # Middle section: Results (with AI insights inside) and CPU Radar side by side - COMPACT
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill="x", padx=20, pady=3)

        # Results Panel with AI Insights (left) - smaller
        self.output = ttk.LabelFrame(middle_frame, text="📊 Results & AI Insights", padding=8)
        self.output.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Combined scrollable text area for results + AI insights
        self.text = tk.Text(self.output, height=8, font=("Consolas", 8), 
                            bg="#fafafa", fg="#2c3e50", relief="flat", padx=6, pady=6)
        self.text.pack(fill="both", expand=True)
        self.text.insert(tk.END, "Click 'Run Optimization' to start...\n\n🤖 AI Insights will appear below results...")
        self.text.config(state="disabled")
        
        # Keep insight_text as hidden reference for compatibility
        self.insight_text = self.text

        # CPU Radar Panel (right) - fixed smaller size
        self.radar_frame = ttk.LabelFrame(middle_frame, text="⚖ CPU Load Balance", padding=5)
        self.radar_frame.pack(side="right", fill="y", padx=(5, 0))
        self.radar_frame.configure(width=280, height=200)
        self.radar_frame.pack_propagate(False)  # Prevent resizing

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
            
            # Setup live convergence chart
            self.setup_live_chart(gmo.max_iters)
            
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
                
                # Update live chart every 2 iterations for smoother animation
                if (i + 1) % 2 == 0 or i == gmo.max_iters - 1:
                    self.update_live_chart()
                
                self.root.update()

            x, f, v = gmo.best_solution()
            _, _, diag = deployable_cloud_objective(x, traffic, n)

            self.status_label.config(text="✓ Complete", foreground="#27ae60")
            self.run_btn.config(state="normal")
            self.show_results(x, f, v, diag, traffic, n)

        except Exception as e:
            self.status_label.config(text="✗ Error", foreground="#e74c3c")
            self.run_btn.config(state="normal")
            messagebox.showerror("Error", str(e))

    def setup_live_chart(self, max_iters):
        """Setup the live convergence chart before optimization starts"""
        # Clear previous charts
        for w in self.plot_frame.winfo_children():
            w.destroy()
        
        # Create figure for live chart
        self.live_fig, self.live_ax = plt.subplots(figsize=(14, 4))
        self.live_fig.patch.set_facecolor('#f0f4f8')
        self.live_ax.set_facecolor('#fafafa')
        
        # Initialize empty line
        self.live_line, = self.live_ax.plot([], [], color='#e74c3c', linewidth=2.5, label='Optimization Cost')
        self.live_fill = None
        
        # Setup axes
        self.live_ax.set_xlim(0, max_iters)
        self.live_ax.set_ylim(0, 1000)  # Will be adjusted dynamically
        self.live_ax.set_title("🔄 Live Optimization Progress", fontsize=14, fontweight='bold', color='#2c3e50', pad=12)
        self.live_ax.set_xlabel("Iteration", fontsize=11, color='#7f8c8d')
        self.live_ax.set_ylabel("Cost", fontsize=11, color='#7f8c8d')
        self.live_ax.tick_params(colors='#7f8c8d', labelsize=10)
        self.live_ax.spines['top'].set_visible(False)
        self.live_ax.spines['right'].set_visible(False)
        self.live_ax.grid(True, alpha=0.3, linestyle='--')
        self.live_ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout(pad=2.0)
        
        # Embed in tkinter
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=self.plot_frame)
        self.live_canvas.draw()
        self.live_canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_live_chart(self):
        """Update the live chart with current iteration data"""
        if not self.iteration_history:
            return
        
        iterations = [h['iteration'] for h in self.iteration_history]
        costs = [h['cost'] for h in self.iteration_history]
        
        # Update line data
        self.live_line.set_data(iterations, costs)
        
        # Update fill area
        if self.live_fill:
            self.live_fill.remove()
        self.live_fill = self.live_ax.fill_between(iterations, costs, alpha=0.3, color='#e74c3c')
        
        # Adjust y-axis dynamically
        if costs:
            max_cost = max(costs)
            min_cost = min(costs)
            padding = (max_cost - min_cost) * 0.1 if max_cost != min_cost else max_cost * 0.1
            self.live_ax.set_ylim(max(0, min_cost - padding), max_cost + padding)
        
        # Update current cost display in title
        current_cost = costs[-1] if costs else 0
        self.live_ax.set_title(f"🔄 Live Optimization Progress  |  Current Cost: {current_cost:.2f}", 
                               fontsize=14, fontweight='bold', color='#2c3e50', pad=12)
        
        # Redraw
        self.live_canvas.draw()

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

        prompt = f"""
Traffic Profile: {traffic}
{explanation}

CPU: {cpu}
RAM: {ram}
Replicas: {rep}

Explain this architecture simply and suggest one improvement.
"""
        
        # Append AI insights to the same text area
        self.text.insert(tk.END, "\n" + "─" * 40 + "\n")
        self.text.insert(tk.END, "🤖 AI INSIGHTS:\n")
        self.text.insert(tk.END, "─" * 40 + "\n")
        self.text.insert(tk.END, "Loading...\n")
        self.root.update()
        
        # Get AI insights and append
        ai_response = get_gemini_insights(prompt)
        # Remove "Loading..." and add actual response
        self.text.delete("end-2l", "end-1l")
        self.text.insert(tk.END, ai_response + "\n")
        
        self.text.config(state="disabled")

        self.show_charts(cpu, ram, rep, n)
        
        # Draw CPU Radar in separate panel
        self.show_cpu_radar_panel(cpu, n)

    def show_charts(self, cpu, ram, rep, n):
        # Clear previous charts
        for w in self.plot_frame.winfo_children():
            w.destroy()
        
        # Create figure with 2 charts: Convergence and Resource Summary
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor('#f0f4f8')
        
        # Use GridSpec for precise control over subplot layout
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], 
                              hspace=0.35,
                              left=0.08, right=0.95, top=0.93, bottom=0.08)
        
        # Row 1: Convergence chart
        ax1 = fig.add_subplot(gs[0])
        
        # Row 2: Resource summary
        ax2 = fig.add_subplot(gs[1])
        
        # Draw convergence chart showing cost over iterations
        if hasattr(self, 'iteration_history') and self.iteration_history:
            iterations = [h['iteration'] for h in self.iteration_history]
            costs = [h['cost'] for h in self.iteration_history]
            
            ax1.set_facecolor('#fafafa')
            ax1.plot(iterations, costs, color='#e74c3c', linewidth=2.5, label='Optimization Cost')
            ax1.fill_between(iterations, costs, alpha=0.3, color='#e74c3c')
            ax1.set_title("Optimization Convergence Over Iterations", fontsize=13, fontweight='bold', color='#2c3e50', pad=12)
            ax1.set_xlabel("Iteration", fontsize=11, color='#7f8c8d')
            ax1.set_ylabel("Cost", fontsize=11, color='#7f8c8d')
            ax1.tick_params(colors='#7f8c8d', labelsize=10)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right', fontsize=10)
        
        # Draw combined resource summary chart
        self.draw_resource_summary(ax2, cpu, ram, rep, n)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add iteration details button
        self.add_iteration_details_panel(n)

    def show_cpu_radar_panel(self, cpu, n):
        """Show CPU Radar chart in the dedicated radar panel"""
        # Clear previous content
        for w in self.radar_frame.winfo_children():
            w.destroy()
        
        # Create smaller figure for radar chart
        fig = plt.figure(figsize=(2.8, 2.5))
        fig.patch.set_facecolor('#f0f4f8')
        
        ax = fig.add_subplot(111, projection='polar')
        self.draw_cpu_radar(ax, cpu, n)
        
        plt.tight_layout(pad=0.5)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.radar_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

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

    def draw_cpu_radar(self, ax, cpu, n):
        """
        Draw CPU Load Balance Radar Chart
        A perfect circle = perfectly balanced system (zero imbalance)
        Spikes show where the system is working hardest
        """
        # Calculate angles for each service
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        
        # Close the polygon by appending the first value
        cpu_values = cpu + [cpu[0]]
        angles += angles[:1]
        
        # Calculate the mean for reference circle (ideal balance)
        mean_cpu = np.mean(cpu)
        mean_values = [mean_cpu] * (n + 1)
        
        # Draw the ideal balance circle (dashed)
        ax.plot(angles, mean_values, color='#27ae60', linewidth=1.5, linestyle='--', 
                label=f'Ideal', alpha=0.7)
        ax.fill(angles, mean_values, color='#27ae60', alpha=0.1)
        
        # Draw the actual CPU allocation
        ax.plot(angles, cpu_values, color='#3498db', linewidth=2, label='Actual')
        ax.fill(angles, cpu_values, color='#3498db', alpha=0.3)
        
        # Set the labels for each service
        service_labels = [f'S{i+1}' for i in range(n)]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(service_labels, fontsize=7, color='#2c3e50')
        
        # Set radial limits
        max_cpu = max(cpu) if cpu else 4.0
        ax.set_ylim(0, max_cpu * 1.2)
        
        # Style the radar chart
        ax.set_facecolor('#fafafa')
        ax.tick_params(colors='#7f8c8d', labelsize=6)
        ax.grid(True, alpha=0.3, linestyle='-')
        
        # Calculate imbalance (standard deviation)
        imbalance = np.std(cpu)
        ax.set_title(f'Imbalance: {imbalance:.3f}', 
                     fontsize=9, fontweight='bold', color='#2c3e50', pad=8)
        
        # Small legend
        ax.legend(loc='upper right', fontsize=6, framealpha=0.8)

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
