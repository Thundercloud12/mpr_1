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
        return 0
    guess = x
    for _ in range(10):
        guess = 0.5 * (guess + x / guess)
    return guess


def exp(x):
    # Taylor approx
    term = 1.0
    result = 1.0
    for i in range(1, 15):
        term *= x / i
        result += term
    return result


def ln(x):
    # Simple log approximation
    n = 100
    result = 0.0
    for i in range(1, n):
        result += (1.0 / i) * ((x - 1) / x) ** i
    return result


def cos(x):
    # Taylor series
    term = 1.0
    result = 1.0
    sign = -1
    for i in range(2, 12, 2):
        term *= x * x / (i * (i - 1))
        result += sign * term
        sign *= -1
    return result


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
    def __init__(self, dim, pop_size, iters, bounds, obj_fn):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = iters
        self.bounds = bounds
        self.obj_fn = obj_fn

        self.rng = RNG()

        self.agents = [
            GMOAgent(dim, bounds, obj_fn, self.rng)
            for _ in range(pop_size)
        ]

        self.current_iteration = 0

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
            mf = 1.0 / (1.0 + exp(-z))
            MF.append(mf)

        return MF

    def compute_dfi(self, MF):
        # log-based stable computation
        log_total = 0.0
        for m in MF:
            log_total += ln(m + 1e-10)

        DFI = []
        for m in MF:
            val = exp(log_total - ln(m + 1e-10))
            DFI.append(val)

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
            return

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

    def run(self):
        for _ in range(self.max_iters):
            self.step()

    def best_solution(self):
        best = self.agents[0]
        for a in self.agents:
            if (a.violation_best < best.violation_best or
               (a.violation_best == best.violation_best and a.f_best < best.f_best)):
                best = a

        return best.x_best, best.f_best, best.violation_best