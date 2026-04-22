import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import math
import numpy as np

# ==========================================
# 1. CORE GMO ALGORITHM (Corrected per paper)
# ==========================================

class GMOAgent:
    def __init__(self, dim, bounds, obj_fn):
        self.dim = dim
        self.bounds = bounds
        self.obj_fn = obj_fn
        
        # Initialize randomly within bounds
        self.x = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        self.v = [0.0 for _ in range(dim)]
        
        self.x_best = self.x[:]
        self.f_best = float('inf')
        self.violation_best = float('inf')
        
        self.evaluate()

    def evaluate(self):
        f, violation = self.obj_fn(self.x)
        
        # Deb's Rules for Constraint Handling
        if violation == 0 and self.violation_best == 0:
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x[:]
        elif violation == 0 and self.violation_best > 0:
            self.f_best = f
            self.violation_best = 0
            self.x_best = self.x[:]
        elif violation > 0 and self.violation_best > 0:
            if violation < self.violation_best:
                self.f_best = f
                self.violation_best = violation
                self.x_best = self.x[:]
                
        return f, violation

    def update_velocity(self, v_new):
        self.v = v_new[:]

    def update_position(self, x_new):
        for d in range(self.dim):
            low, high = self.bounds[d]
            val = max(low, min(high, x_new[d])) # Clamp to bounds
            self.x[d] = val

class GMO:
    def __init__(self, dim, pop_size, iters, bounds, obj_fn):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = iters
        self.bounds = bounds
        self.obj_fn = obj_fn
        self.current_iteration = 0
        
        self.agents = [GMOAgent(dim, bounds, obj_fn) for _ in range(pop_size)]

    def compute_statistics(self):
        fitness = []
        for a in self.agents:
            # Penalize highly if constraints are violated
            fit = a.f_best if a.violation_best == 0 else a.f_best + 1e6 * a.violation_best
            fitness.append(fit)
            
        mu = sum(fitness) / self.pop_size
        var = sum((f - mu)**2 for f in fitness) / self.pop_size
        sigma = math.sqrt(var) + 1e-10
        return mu, sigma

    def compute_membership(self, mu, sigma):
        MF = []
        # Incorporating sqrt(e) ≈ 1.64872127 from the paper
        a_param = -4.0 / (sigma * 1.64872127 + 1e-10) 
        for agent in self.agents:
            z = a_param * (agent.f_best - mu)
            try:
                mf = 1.0 / (1.0 + math.exp(-z))
            except OverflowError:
                mf = 0.0 if z < 0 else 1.0
            MF.append(mf)
        return MF

    def compute_dfi(self, MF):
        log_total = sum(math.log(m + 1e-10) for m in MF)
        DFI = [math.exp(log_total - math.log(m + 1e-10)) for m in MF]
        return DFI

    def select_elites(self, DFI):
        idx = list(range(self.pop_size))
        idx.sort(key=lambda i: (self.agents[i].violation_best, -DFI[i]))
        
        # Linearly decreasing N_best
        fraction_complete = self.current_iteration / max(1, self.max_iters - 1)
        k = int(self.pop_size - fraction_complete * (self.pop_size - 2))
        k = max(2, min(self.pop_size, k))
        return idx[:k]

    def generate_guides(self, DFI, elites):
        guides = []
        for i in range(self.pop_size):
            sum_w = sum(DFI[e] for e in elites if self.agents[e].violation_best == 0)
            
            if sum_w == 0: # Fallback if all elites violate constraints
                sum_w = sum(DFI[e] for e in elites)
                
            guide = [0.0] * self.dim
            for e in elites:
                w = DFI[e]
                for d in range(self.dim):
                    guide[d] += w * self.agents[e].x_best[d]
                    
            if sum_w == 0:
                guides.append(self.agents[i].x[:]) 
            else:
                guides.append([g / sum_w for g in guide])
        return guides

    def mutate(self, guides, w):
        mutated = []
        std_dim = []
        
        # Dimension-wise standard deviation for Gaussian mutation
        for d in range(self.dim):
            vals = [a.x_best[d] for a in self.agents]
            mean_val = sum(vals) / self.pop_size
            var_val = sum((v - mean_val)**2 for v in vals) / self.pop_size
            std_dim.append(math.sqrt(var_val))
            
        std_max = max(std_dim) if std_dim else 0.0

        for g in guides:
            new = []
            for d in range(self.dim):
                noise = w * random.gauss(0, 1) * (std_max - std_dim[d])
                new.append(g[d] + noise)
            mutated.append(new)
        return mutated

    def update(self, guides, w):
        for i in range(self.pop_size):
            new_v = []
            new_x = []
            for d in range(self.dim):
                phi = 1.0 + (2 * random.random() - 1.0) * w
                v = w * self.agents[i].v[d] + phi * (guides[i][d] - self.agents[i].x[d])
                x = self.agents[i].x[d] + v
                new_v.append(v)
                new_x.append(x)
                
            self.agents[i].update_velocity(new_v)
            self.agents[i].update_position(new_x)

    def step(self):
        if self.current_iteration >= self.max_iters: return False
        
        w = 1.0 - (self.current_iteration / self.max_iters)
        mu, sigma = self.compute_statistics()
        MF = self.compute_membership(mu, sigma)
        DFI = self.compute_dfi(MF)
        
        elites = self.select_elites(DFI)
        guides = self.generate_guides(DFI, elites)
        guides = self.mutate(guides, w)
        self.update(guides, w)
        
        for a in self.agents:
            a.evaluate()
            
        self.current_iteration += 1
        return True

    def best_solution(self):
        best = self.agents[0]
        for a in self.agents:
            if (a.violation_best < best.violation_best or 
               (a.violation_best == best.violation_best and a.f_best < best.f_best)):
                best = a
        return best.x_best, best.f_best, best.violation_best


# ==========================================
# 2. GUI DASHBOARD (Tkinter + Matplotlib)
# ==========================================

class GMODashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("GMO Optimization Dashboard")
        self.root.geometry("1000x700")
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="Step-by-Step Visualization (1D)")
        self.notebook.add(self.tab2, text="Custom Function Minimizer (N-D)")
        
        self.setup_tab1()
        self.setup_tab2()

    # ------------------- TAB 1: 1D Visualization -------------------
    def setup_tab1(self):
        # Controls frame
        control_frame = ttk.Frame(self.tab1, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(control_frame, text="Function: f(x) = x²").pack(side=tk.LEFT, padx=10)
        self.btn_reset = ttk.Button(control_frame, text="Reset / Initialize", command=self.init_1d_gmo)
        self.btn_reset.pack(side=tk.LEFT, padx=5)
        
        self.btn_step = ttk.Button(control_frame, text="Next Step", command=self.step_1d_gmo, state=tk.DISABLED)
        self.btn_step.pack(side=tk.LEFT, padx=5)
        
        self.lbl_iter = ttk.Label(control_frame, text="Iteration: 0")
        self.lbl_iter.pack(side=tk.LEFT, padx=20)
        
        # Plot frame
        self.fig1, self.ax1 = plt.subplots(figsize=(8, 5))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Base function logic for Tab 1
        self.func_1d = lambda x: (x[0]**2, 0) # Returns (fitness, violation)
        self.gmo_1d = None

    def init_1d_gmo(self):
        # Initialize GMO for 1D problem
        bounds = [(-10, 10)]
        self.gmo_1d = GMO(dim=1, pop_size=15, iters=50, bounds=bounds, obj_fn=self.func_1d)
        
        self.btn_step.config(state=tk.NORMAL)
        self.update_1d_plot()

    def step_1d_gmo(self):
        if self.gmo_1d and self.gmo_1d.step():
            self.update_1d_plot()
        else:
            self.btn_step.config(state=tk.DISABLED)
            messagebox.showinfo("Done", "Maximum iterations reached.")

    def update_1d_plot(self):
        self.ax1.clear()
        
        # Plot the landscape
        X = np.linspace(-10, 10, 200)
        Y = X**2
        self.ax1.plot(X, Y, label="f(x) = x²", color='blue', alpha=0.5)
        
        # Plot agents
        agent_x = [a.x[0] for a in self.gmo_1d.agents]
        agent_y = [a.x[0]**2 for a in self.gmo_1d.agents]
        
        self.ax1.scatter(agent_x, agent_y, color='red', s=50, zorder=5, label="Agents")
        
        # Highlight best
        best_x, _, _ = self.gmo_1d.best_solution()
        self.ax1.scatter([best_x[0]], [best_x[0]**2], color='gold', edgecolor='black', s=150, marker='*', zorder=6, label="Global Best")
        
        self.ax1.set_title(f"GMO 1D Optimization (Iteration {self.gmo_1d.current_iteration})")
        self.ax1.set_xlim(-10, 10)
        self.ax1.set_ylim(-5, 105)
        self.ax1.legend()
        
        self.lbl_iter.config(text=f"Iteration: {self.gmo_1d.current_iteration} / {self.gmo_1d.max_iters}")
        self.canvas1.draw()

    # ------------------- TAB 2: Custom Function Minimizer -------------------
    def setup_tab2(self):
        left_frame = ttk.Frame(self.tab2, padding=10, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        right_frame = ttk.Frame(self.tab2, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Settings
        ttk.Label(left_frame, text="Function to minimize (Python syntax):").pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(left_frame, text="Use x1, x2, x3... for variables", font=("Arial", 8, "italic")).pack(anchor=tk.W)
        self.entry_func = ttk.Entry(left_frame, width=40)
        self.entry_func.insert(0, "x1**2 + math.sin(x2)")
        self.entry_func.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(left_frame, text="Dimensions:").pack(anchor=tk.W)
        self.entry_dim = ttk.Entry(left_frame, width=10)
        self.entry_dim.insert(0, "2")
        self.entry_dim.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(left_frame, text="Bounds (comma separated, e.g., -10,10):").pack(anchor=tk.W)
        self.entry_bounds = ttk.Entry(left_frame, width=20)
        self.entry_bounds.insert(0, "-10,10")
        self.entry_bounds.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(left_frame, text="Population Size:").pack(anchor=tk.W)
        self.entry_pop = ttk.Entry(left_frame, width=10)
        self.entry_pop.insert(0, "30")
        self.entry_pop.pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(left_frame, text="Iterations:").pack(anchor=tk.W)
        self.entry_iters = ttk.Entry(left_frame, width=10)
        self.entry_iters.insert(0, "100")
        self.entry_iters.pack(anchor=tk.W, pady=(0, 20))
        
        ttk.Button(left_frame, text="Run Optimization", command=self.run_custom_gmo).pack(fill=tk.X, pady=5)
        
        # Results text
        self.txt_results = tk.Text(left_frame, height=12, width=35)
        self.txt_results.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Plot setup
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_custom_gmo(self):
        try:
            dim = int(self.entry_dim.get())
            pop = int(self.entry_pop.get())
            iters = int(self.entry_iters.get())
            b_min, b_max = map(float, self.entry_bounds.get().split(','))
            bounds = [(b_min, b_max) for _ in range(dim)]
            func_str = self.entry_func.get()
            
            # Secure-ish evaluation wrapper
            def custom_eval(x_list):
                local_dict = {f"x{i+1}": val for i, val in enumerate(x_list)}
                local_dict.update({k: v for k, v in math.__dict__.items() if not k.startswith('_')})
                try:
                    res = eval(func_str, {"__builtins__": None}, local_dict)
                    return float(res), 0 # Return fitness and 0 violation (no constraints in this basic tab)
                except Exception as e:
                    raise ValueError(f"Function evaluation error: {e}")

            # Initialize and run
            gmo = GMO(dim=dim, pop_size=pop, iters=iters, bounds=bounds, obj_fn=custom_eval)
            
            history = []
            for _ in range(iters):
                gmo.step()
                _, f, _ = gmo.best_solution()
                history.append(f)
                
            best_x, best_f, _ = gmo.best_solution()
            
            # Update UI text
            self.txt_results.delete(1.0, tk.END)
            self.txt_results.insert(tk.END, "--- Optimization Complete ---\n\n")
            for i, val in enumerate(best_x):
                self.txt_results.insert(tk.END, f"x{i+1} = {val:.6f}\n")
            self.txt_results.insert(tk.END, f"\nMinimum Cost:\n{best_f:.6e}")
            
            # Update plot
            self.ax2.clear()
            self.ax2.plot(history, color='green', linewidth=2)
            self.ax2.set_title("Convergence History")
            self.ax2.set_xlabel("Iteration")
            self.ax2.set_ylabel("Best Cost")
            self.ax2.grid(True, linestyle='--', alpha=0.7)
            self.canvas2.draw()

        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = GMODashboard(root)
    root.mainloop()