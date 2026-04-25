import math
from dataclasses import dataclass
from typing import Callable, Sequence

ObjectiveFn = Callable[[Sequence[float]], tuple[float, int]]


@dataclass(frozen=True)
class PresetDefinition:
    dim: int
    bounds: str
    expression: str
    fn: ObjectiveFn


def sphere_objective(x: Sequence[float]) -> tuple[float, int]:
    return sum(v * v for v in x), 0


def rosenbrock_objective(x: Sequence[float]) -> tuple[float, int]:
    if len(x) < 2:
        return sphere_objective(x)

    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2
    return total, 0


def rastrigin_objective(x: Sequence[float]) -> tuple[float, int]:
    n = len(x)
    total = 10.0 * n
    for v in x:
        total += v * v - 10.0 * math.cos(2.0 * math.pi * v)
    return total, 0


def ackley_objective(x: Sequence[float]) -> tuple[float, int]:
    n = len(x)
    if n == 0:
        return 0.0, 0

    sum_sq = sum(v * v for v in x)
    sum_cos = sum(math.cos(2.0 * math.pi * v) for v in x)

    term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / n))
    term2 = -math.exp(sum_cos / n)
    return term1 + term2 + 20.0 + math.e, 0


PRESET_CONFIG: dict[str, PresetDefinition] = {
    "Sphere": PresetDefinition(
        dim=2,
        bounds="-5.12,5.12;-5.12,5.12",
        expression="sum(v*v for v in x)",
        fn=sphere_objective,
    ),
    "Rosenbrock": PresetDefinition(
        dim=2,
        bounds="-3,3;-3,3",
        expression="sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))",
        fn=rosenbrock_objective,
    ),
    "Rastrigin": PresetDefinition(
        dim=2,
        bounds="-5.12,5.12;-5.12,5.12",
        expression="10*len(x) + sum(v*v - 10*math.cos(2*math.pi*v) for v in x)",
        fn=rastrigin_objective,
    ),
    "Ackley": PresetDefinition(
        dim=2,
        bounds="-32.768,32.768;-32.768,32.768",
        expression="-20*exp(-0.2*sqrt(sum(v*v for v in x)/len(x))) - exp(sum(cos(2*pi*v) for v in x)/len(x)) + 20 + e",
        fn=ackley_objective,
    ),
}

PRESET_NAMES = [*PRESET_CONFIG.keys(), "Custom"]
