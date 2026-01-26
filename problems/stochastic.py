import random
import math
import numpy as np

def stochastic_rastrigin(x, sigma=0.4):
    """
    Expectation-defined objective.
    True global minimum at x = [-2.5]
    Noise is temporally unstable but spatially smooth.
    Returns (objective, constraint_violation)
    """

    # True deterministic landscape
    z = x[0] + 2.5
    base = z**2 - 10 * math.cos(2 * math.pi * z) + 10

    # Spatially smooth noise (correlated in x)
    spatial_noise = sigma * math.sin(6 * x[0])

    # Temporal noise (breaks DE selection)
    temporal_noise = random.gauss(0, sigma * 0.3)

    f = base + spatial_noise + temporal_noise
    
    # No constraints in this problem
    violation = 0
    
    return f, violation
