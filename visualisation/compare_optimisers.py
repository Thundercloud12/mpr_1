import numpy as np
import matplotlib.pyplot as plt

from pso import PSO
from gmo import GMO


# =========================
# Objective Function
# =========================
def rastrigin(x):
    return x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10


# =========================
# Experiment Config
# =========================
BOUNDS = [(-5.12, 5.12)]
POP_SIZE = 30
ITERATIONS = 200
TRIALS = 20   # important for fair comparison


# =========================
# Run Experiments
# =========================
def run_experiment(optimizer_class, name):
    all_runs = []

    for trial in range(TRIALS):
        opt = optimizer_class(
            obj_func=rastrigin,
            bounds=BOUNDS,
            pop_size=POP_SIZE,
            iterations=ITERATIONS
        )

        best_history = opt.optimize()  # expects list of best fitness per iteration
        all_runs.append(best_history)

        print(f"{name} | Trial {trial+1}/{TRIALS} | Final best = {best_history[-1]:.6f}")

    return np.array(all_runs)


print("\nRunning PSO...")
pso_runs = run_experiment(PSO, "PSO")

print("\nRunning GMO...")
gmo_runs = run_experiment(GMO, "GMO")


# =========================
# Statistics
# =========================
pso_mean = pso_runs.mean(axis=0)
pso_std  = pso_runs.std(axis=0)

gmo_mean = gmo_runs.mean(axis=0)
gmo_std  = gmo_runs.std(axis=0)


# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))

plt.plot(pso_mean, label="PSO (mean)", linewidth=2)
plt.fill_between(
    range(ITERATIONS),
    pso_mean - pso_std,
    pso_mean + pso_std,
    alpha=0.2
)

plt.plot(gmo_mean, label="GMO (mean)", linewidth=2)
plt.fill_between(
    range(ITERATIONS),
    gmo_mean - gmo_std,
    gmo_mean + gmo_std,
    alpha=0.2
)

plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("PSO vs GMO on 1D Rastrigin Function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
