import math

# constants
P = 6000
L = 14
E = 30e6
G = 12e6

tau_max = 13600
sigma_max = 30000
delta_max = 0.25


def welded_beam(x):
    """
    Returns (objective, total_constraint_violation)
    x = [h, l, t, b]
    """

    h, l, t, b = x

    # Objective
    f = 1.10471 * h**2 * l + 0.04811 * t * b * (14 + l)

    # ---- Stress calculations ----
    tau_p = P / (math.sqrt(2) * h * l)

    M = P * (L + l / 2)
    R = math.sqrt(l**2 / 4 + (h + t)**2 / 4)

    J = 2 * (math.sqrt(2) * h * l *
             (l**2 / 12 + (h + t)**2 / 4))

    tau_pp = M * R / J
    tau = math.sqrt(tau_p**2 + 2 * tau_p * tau_pp * l / (2 * R) + tau_pp**2)

    sigma = 6 * P * L / (b * t**2)
    delta = 4 * P * L**3 / (E * t**3 * b)

    Pc = (4.013 * E * math.sqrt(t**2 * b**6 / 36) / L**2) * \
         (1 - t / (2 * L) * math.sqrt(E / (4 * G)))

    # ---- Constraints g(x) ≤ 0 ----
    g = [
        tau - tau_max,
        sigma - sigma_max,
        h - b,
        delta - delta_max,
        P - Pc,
        0.125 - h
    ]

    violation = sum(max(0, gi) for gi in g)

    return f, violation
