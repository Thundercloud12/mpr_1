from algorithms.pso import PSO
from algorithms.gmo import GMO
from algorithms.de import DE
from visualisation.visualize_1d import Optimizer1DVisualizer
import random
import numpy as np

import math

def rastrigin(x):
    return x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10


def noisy_rastrigin(x):
    return rastrigin(x) + random.gauss(0, 0.1)

def controlled_noisy_rastrigin(x, sigma=0.5):
    """
    Temporally unstable but spatially smooth noise
    Breaks PSO memory, preserves GMO statistics
    """

    base = 10 * len(x)
    base += sum(
        xi**2 - 10 * np.cos(2 * np.pi * xi)
        for xi in x
    )

    # smooth spatial noise (same nearby x → similar noise)
    spatial_noise = sigma * np.sin(5 * x[0])

    # small random jitter (NOT heavy-tailed)
    temporal_noise = random.gauss(0, sigma * 0.15)

    return base + spatial_noise + temporal_noise

def simple_shifted_quadratic(x):
    """
    Simple 1D convex function
    Global minimum is NOT zero
    """

    return (x[0] - 2)**2 + 3


def sinusoidal_with_unique_global_min(x):
    """
    Multiple local minima (sinusoidal ripples)
    One unique global minimum
    Global minimum is NOT at zero
    """

    return (np.sin(1/x[0]))

def shifted_rastrigin(x, shift=-2.5):
    """
    Rastrigin function with global minimum shifted to the left
    """

    z = x[0] - shift
    return z**2 - 10 * np.cos(2 * np.pi * z) + 10


def rosenbrock(x, a=1, b=100):
    """
    Rosenbrock function (banana function)
    Global minimum at x = [a, a^2]
    Minimum value = 0
    """

    return (a - x[0])**2 + b * (10 - x[0]**2)**2

def de_breaker_gmo_survivor(x, sigma=0.4):
    """
    Designed to break Differential Evolution
    but remain solvable by GMO-style averaging methods
    """

    # Global convex basin (truth only visible in expectation)
    base = (x[0] + 2.0)**2 + 5

    # High-frequency deception: breaks DE difference vectors
    deceptive_ripples = 1.5 * np.sin(18 * x[0])

    # Spatially smooth noise (correlated in x)
    spatial_noise = sigma * np.sin(7 * x[0] + 0.3)

    # Small temporal noise (kills pairwise comparisons)
    temporal_noise = random.gauss(0, sigma * 0.2)

    return base + deceptive_ripples + spatial_noise + temporal_noise


bounds = [(-5, 5)]

pso =GMO(
    dim=1,
    pop_size=40,
    iters=100,
    bounds=bounds,
    obj_fn=de_breaker_gmo_survivor
)

pso.run()

viz = Optimizer1DVisualizer(
    history_positions=pso.history_positions,
    obj_fn=de_breaker_gmo_survivor,
    bounds=bounds,
    title="PSO on f(x) = x²"
)



viz.animate()
