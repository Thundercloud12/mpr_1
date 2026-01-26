import numpy as np
import random
import matplotlib.pyplot as plt

from algorithms.gmo import GMO
from algorithms.pso import PSO
from algorithms.de import DE

# ==============================
# Problem Configuration
# ==============================
DIM = 50                  # Resource units
BOUNDS = [(0.0, 1.0)] * DIM
TOTAL_RESOURCE = 60

     # System capacity
NOISE_STD = 0.15
MIN_UTILIZATION = 0.4 * TOTAL_RESOURCE  # 48


# ==============================
# Objective Function
# ==============================
def resource_allocation_objective(x):
    x = np.array(x)

    total_used = x.sum()
    imbalance = np.std(x)
    geometric_balance = np.prod(x + 1e-6)**(1 / DIM)
    utilization_reward = (total_used / TOTAL_RESOURCE)


    # Hard feasibility constraints
    overuse_violation = max(0, total_used - TOTAL_RESOURCE)
    underuse_violation = max(0, MIN_UTILIZATION - total_used)

    violation = overuse_violation + underuse_violation

    # Soft stability penalties
    imbalance_penalty = imbalance ** 2

    noise = random.gauss(0, NOISE_STD)

    f = (
        -geometric_balance
        - 0.5 * utilization_reward 
        + 0.8 * imbalance_penalty
        + 5.0 * violation**2
        + noise
    )

    return f, violation


# ==============================
# Run Optimizers
# ==============================
def run_optimizer(optimizer_class, name, dim, pop_size, iters, bounds, obj_fn):
    """
    Initialize and run optimizer, return best fitness history.
    
    Args:
        optimizer_class: PSO, DE, or GMO class
        name: Algorithm name (for display)
        dim: Problem dimensionality
        pop_size: Population size
        iters: Number of iterations
        bounds: Search space bounds
        obj_fn: Objective function
    
    Returns:
        history: Best fitness per iteration
    """
    optimizer = optimizer_class(
        dim=dim,
        pop_size=pop_size,
        iters=iters,
        bounds=bounds,
        obj_fn=obj_fn
    )
    
    optimizer.run()  # Execute optimization
    x, f, v = optimizer.best_solution()  # Get best solution

    x = np.array(x)

    total = x.sum()
    mean = x.mean()
    std = x.std()
    cv = std / mean

    print(f"Total Used Capacity: {total:.2f}")
    print(f"Min Allocation:      {x.min():.2f}")
    print(f"Max Allocation:      {x.max():.2f}")
    print(f"Mean Allocation:     {mean:.2f}")
    print(f"Std Deviation:       {std:.2f}")
    print(f"Coeff of Variation:  {cv:.2f}")

    
    print(f"{name:5} | Best Objective: {f:10.6f} | Constraint Violation: {v:10.6e}")
    
    return optimizer.history_best

# ==============================
# Execute Experiments
# ==============================
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("RESOURCE ALLOCATION OPTIMIZATION (Stability Under Noise)")
    print("="*70 + "\n")
    
    print(f"Configuration:")
    print(f"  Dimensions:      {DIM}")
    print(f"  Population:      40")
    print(f"  Iterations:      150")
    print(f"  Noise Std Dev:   {NOISE_STD}")
    print("\n" + "-"*70)
    print("Algorithm | Best Objective | Constraint Violation")
    print("-"*70 + "\n")
    
    # Run all three optimizers
    gmo_hist = run_optimizer(
        GMO, "GMO", DIM, 40, 150, BOUNDS, resource_allocation_objective
    )
    
    pso_hist = run_optimizer(
        PSO, "PSO", DIM, 40, 150, BOUNDS, resource_allocation_objective
    )
    
    de_hist = run_optimizer(
        DE, "DE", DIM, 40, 150, BOUNDS, resource_allocation_objective
    )
    
    print("\n" + "="*70 + "\n")
    
    # ==============================
    # Plot Convergence
    # ==============================
    plt.figure(figsize=(12, 6))
    
    plt.plot(gmo_hist, label="GMO", linewidth=2.5, color="blue")
    plt.plot(pso_hist, label="PSO", linewidth=2.5, color="red", linestyle="--")
    plt.plot(de_hist, label="DE", linewidth=2.5, color="green", linestyle=":")
    
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Objective Value", fontsize=12)
    plt.title("Resource Allocation Optimization\n(Stochastic Fitness, 20D)", fontsize=14)
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
