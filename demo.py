import numpy as np
import random
import statistics
import matplotlib.pyplot as plt

from algorithms.pso import PSO
from algorithms.de import DE
from algorithms.gmo import GMO

# =====================================================
# High-Dimensional Stochastic Objective (50D)
# =====================================================
def stochastic_rastrigin_35d(x):
    """
    35-D Rastrigin with dimension-scaled noise and weak variable coupling.
    Designed to expose PSO instability under stochastic fitness.
    Returns (objective, constraint_violation)
    """
    A = 10
    x_arr = np.array(x)
    n = 20  # fixed dimensionality

    # Base Rastrigin
    base = A * n + np.sum(x_arr**2 - A * np.cos(2 * np.pi * x_arr))

    # Dimension-scaled stochastic noise
    noise = random.gauss(0, 0.12 * np.sqrt(n))

    # Weak coupling term (breaks separability, hurts PSO)
    coupling = 0.05 * np.sum(x_arr[:-1] * x_arr[1:])

    f = base + coupling + noise
    violation = 0  # unconstrained

    return f, violation
# =====================================================
# CONFIG
# =====================================================
DIM = 50
BOUNDS = [(-6, 6)] * DIM
POP = 60
ITERS = 300
RUNS = 30

ALGOS = {
    "PSO": PSO,
    "DE": DE,
    "GMO": GMO
}


# =====================================================
# BENCHMARK
# =====================================================
def benchmark(Algo):
    best_vals = []
    final_positions = []
    convergence_curves = []

    for _ in range(RUNS):
        algo = Algo(
            dim=DIM,
            pop_size=POP,
            iters=ITERS,
            bounds=BOUNDS,
            obj_fn=stochastic_rastrigin_35d
        )

        history = algo.run()          # MUST return best fitness per iteration
        x, f, v = algo.best_solution()

        best_vals.append(f)
        final_positions.append(x)
        convergence_curves.append(history)

    return {
        "mean_f": statistics.mean(best_vals),
        "std_f": statistics.stdev(best_vals),
        "best_f": min(best_vals),
        "curves": np.array(convergence_curves)
    }


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    print("\n=== HIGH-DIMENSIONAL STOCHASTIC BENCHMARK (50D) ===\n")

    results = {}

    for name, Algo in ALGOS.items():
        print(f"Running {name}...")
        results[name] = benchmark(Algo)

    # =============================
    # PRINT STATS
    # =============================
    print("\nAlgo | Mean f | Std f | Best f")
    print("-" * 45)

    for k, v in results.items():
        print(
            f"{k:<4} | "
            f"{v['mean_f']:.3e} | "
            f"{v['std_f']:.3e} | "
            f"{v['best_f']:.3e}"
        )

    winner = min(results.items(), key=lambda x: x[1]["mean_f"])
    print(f"\n🏆 WINNER (by mean fitness): {winner[0]}")

    # =============================
    # CONVERGENCE PLOT
    # =============================
    plt.figure(figsize=(10, 6))

    for name, v in results.items():
        mean_curve = np.mean(v["curves"], axis=0)
        plt.plot(mean_curve, label=name)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Log Fitness")
    plt.title("50D Stochastic Rastrigin – Mean Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
