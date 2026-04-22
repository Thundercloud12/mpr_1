import math


class RNG:
    def __init__(self, seed=123456):
        self.seed = seed

    def rand(self):
        # Linear Congruential Generator
        self.seed = (1103515245 * self.seed + 12345) % (2**31)
        return self.seed / (2**31)

    def uniform(self, a, b):
        return a + (b - a) * self.rand()

    def gauss(self):
        # Box-Muller (approx, using own log)
        u1 = self.rand() + 1e-10
        u2 = self.rand() + 1e-10
        return sqrt(-2 * ln(u1)) * cos(2 * 3.1415926535 * u2)


# ---- BASIC MATH APPROX ----

def abs_val(x):
    return x if x >= 0 else -x


def sqrt(x):
    if x <= 0:
        return 0.0
    return math.sqrt(x)


def exp(x):
    # Clamp to avoid platform-dependent overflow in extreme exponentials.
    if x > 700:
        x = 700
    elif x < -700:
        x = -700
    return math.exp(x)


def ln(x):
    # Keep away from log(0) while preserving order for tiny values.
    return math.log(max(x, 1e-300))


def cos(x):
    return math.cos(x)


# --------- GMO AGENT ---------

class GMOAgent:
    def __init__(self, dim, bounds, obj_fn, rng):
        self.dim = dim
        self.bounds = bounds
        self.obj_fn = obj_fn
        self.rng = rng

        self.x = [rng.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        self.v = [0.0 for _ in range(dim)]

        self.x_best = self.x[:]
        self.f_best = 1e18
        self.violation_best = 1e18

        self.evaluate()

    def evaluate(self):
        f, violation = self.obj_fn(self.x)

        # Deb's rules
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
            val = x_new[d]
            if val < low:
                val = low
            if val > high:
                val = high
            self.x[d] = val


# --------- GMO ---------

class GMO:
    def __init__(self, dim, pop_size, iters, bounds, obj_fn, seed=123456, track_history=True):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = iters
        self.bounds = bounds
        self.obj_fn = obj_fn
        self.seed = seed
        self.track_history = track_history

        self.rng = RNG(seed)

        self.agents = [
            GMOAgent(dim, bounds, obj_fn, self.rng)
            for _ in range(pop_size)
        ]

        self.current_iteration = 0
        self.history = []
        self.last_metrics = None

    def _feasible_count(self):
        return sum(1 for a in self.agents if a.violation_best == 0)

    def _population_diversity(self):
        if self.pop_size == 0 or self.dim == 0:
            return 0.0

        total_std = 0.0
        for d in range(self.dim):
            mean_d = sum(a.x[d] for a in self.agents) / self.pop_size
            var_d = 0.0
            for a in self.agents:
                diff = a.x[d] - mean_d
                var_d += diff * diff
            var_d /= self.pop_size
            total_std += sqrt(var_d)

        return total_std / self.dim

    def _mean_abs_velocity(self):
        if self.pop_size == 0 or self.dim == 0:
            return 0.0

        total = 0.0
        for a in self.agents:
            for v in a.v:
                total += abs_val(v)

        return total / (self.pop_size * self.dim)

    def get_history(self):
        return list(self.history)

    def get_last_metrics(self):
        if self.last_metrics is None:
            return None
        return dict(self.last_metrics)

    def compute_statistics(self):
        fitness = []
        for a in self.agents:
            if a.violation_best == 0:
                fitness.append(a.f_best)
            else:
                fitness.append(a.f_best + 1e6 * a.violation_best)

        mu = sum(fitness) / self.pop_size

        var = 0.0
        for f in fitness:
            var += (f - mu) * (f - mu)
        var /= self.pop_size

        sigma = sqrt(var) + 1e-10
        return mu, sigma

    def compute_membership(self, mu, sigma):
        MF = []
        a_param = -4.0 / (sigma + 1e-10)

        for agent in self.agents:
            val = agent.f_best
            z = a_param * (val - mu)

            # Numerically stable sigmoid.
            if z >= 0:
                mf = 1.0 / (1.0 + exp(-z))
            else:
                ez = exp(z)
                mf = ez / (1.0 + ez)

            MF.append(mf)

        return MF

    def compute_dfi(self, MF):
        # log-based stable computation
        log_total = 0.0
        for m in MF:
            log_total += ln(m + 1e-10)

        # DFI is only used relatively (ranking/weights), so we can safely
        # shift the exponent by a constant to keep values in range.
        log_vals = [log_total - ln(m + 1e-10) for m in MF]
        max_log = max(log_vals)

        DFI = []
        for lv in log_vals:
            DFI.append(exp(lv - max_log))

        return DFI

    def select_elites(self, DFI):
        idx = list(range(self.pop_size))

        idx.sort(key=lambda i: (
            self.agents[i].violation_best,
            -DFI[i]
        ))

        k = max(2, self.pop_size // 3)
        return idx[:k]

    def generate_guides(self, DFI, elites):
        guides = []

        for i in range(self.pop_size):
            sum_w = 0.0
            guide = [0.0] * self.dim

            for e in elites:
                if self.agents[e].violation_best > 0:
                    continue

                w = DFI[e]
                sum_w += w

                for d in range(self.dim):
                    guide[d] += w * self.agents[e].x_best[d]

            if sum_w == 0:
                guides.append(self.agents[i].x[:])  # fallback
            else:
                guides.append([g / sum_w for g in guide])

        return guides

    def mutate(self, guides, w):
        mutated = []

        for g in guides:
            new = []
            for d in range(self.dim):
                noise = (self.rng.rand() - 0.5) * w
                new.append(g[d] + noise)
            mutated.append(new)

        return mutated

    def update(self, guides, w):
        for i in range(self.pop_size):
            new_v = []
            new_x = []

            for d in range(self.dim):
                phi = 1.0 + (2 * self.rng.rand() - 1.0) * w

                v = w * self.agents[i].v[d] + phi * (guides[i][d] - self.agents[i].x[d])
                x = self.agents[i].x[d] + v

                new_v.append(v)
                new_x.append(x)

            self.agents[i].update_velocity(new_v)
            self.agents[i].update_position(new_x)

    def step(self):
        if self.current_iteration >= self.max_iters:
            return False

        denom = max(1, self.max_iters)
        w = 1.0 - (self.current_iteration / denom)

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

        best_x, best_f, best_v = self.best_solution()
        feasible_count = self._feasible_count()
        metrics = {
            "iteration": self.current_iteration,
            "w": w,
            "mu": mu,
            "sigma": sigma,
            "best_f": best_f,
            "best_violation": best_v,
            "elite_count": len(elites),
            "feasible_count": feasible_count,
            "feasible_ratio": feasible_count / max(1, self.pop_size),
            "diversity": self._population_diversity(),
            "mean_abs_velocity": self._mean_abs_velocity(),
            "best_x": best_x[:],
        }

        self.last_metrics = metrics
        if self.track_history:
            self.history.append(metrics)

        return True

    def run(self):
        while self.step():
            pass

    def best_solution(self):
        best = self.agents[0]
        for a in self.agents:
            if (a.violation_best < best.violation_best or
               (a.violation_best == best.violation_best and a.f_best < best.f_best)):
                best = a

        return best.x_best, best.f_best, best.violation_best