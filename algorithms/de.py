import random

# -------------------------------
# Helper: feasibility-first compare
# -------------------------------
def better(f1, v1, f2, v2):
    if v1 == 0 and v2 == 0:
        return f1 < f2
    if v1 == 0:
        return True
    if v2 == 0:
        return False
    return v1 < v2


class DEAgent:
    def __init__(self, dim, bounds, obj_fn):
        self.x = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        self.f, self.violation = obj_fn(self.x)


class DE:
    def __init__(self, dim, pop_size, iters, bounds, obj_fn, F=0.8, CR=0.9):
        self.dim = dim
        self.pop_size = pop_size
        self.iters = iters
        self.bounds = bounds
        self.obj_fn = obj_fn

        self.F = F
        self.CR = CR

        self.agents = [DEAgent(dim, bounds, obj_fn) for _ in range(pop_size)]

        best = self.agents[0]
        for agent in self.agents[1:]:
            if better(agent.f, agent.violation, best.f, best.violation):
                best = agent

        self.gbest_x = best.x[:]
        self.gbest_f = best.f
        self.gbest_violation = best.violation

        self.history_positions = []
        self.history_best = []

    # -------------------------------
    def evaluate(self):
        for agent in self.agents:
            agent.f, agent.violation = self.obj_fn(agent.x)

            if better(agent.f, agent.violation,
                      self.gbest_f, self.gbest_violation):
                self.gbest_x = agent.x[:]
                self.gbest_f = agent.f
                self.gbest_violation = agent.violation

    # -------------------------------
    def mutate(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = random.sample(indices, 3)

        xa = self.agents[a].x
        xb = self.agents[b].x
        xc = self.agents[c].x

        mutant = [
            xa[d] + self.F * (xb[d] - xc[d])
            for d in range(self.dim)
        ]

        return mutant

    # -------------------------------
    def crossover(self, target, mutant):
        trial = []
        j_rand = random.randrange(self.dim)

        for d in range(self.dim):
            if random.random() < self.CR or d == j_rand:
                trial.append(mutant[d])
            else:
                trial.append(target[d])

        return trial

    # -------------------------------
    def select(self, agent, trial):
        for d in range(self.dim):
            low, high = self.bounds[d]
            trial[d] = max(min(trial[d], high), low)

        f_trial, v_trial = self.obj_fn(trial)

        if better(f_trial, v_trial, agent.f, agent.violation):
            agent.x = trial
            agent.f = f_trial
            agent.violation = v_trial

    # -------------------------------
    def step(self, t):
        for i, agent in enumerate(self.agents):
            mutant = self.mutate(i)
            trial = self.crossover(agent.x, mutant)
            self.select(agent, trial)

        self.evaluate()

        self.history_positions.append([agent.x[:] for agent in self.agents])
        self.history_best.append(self.gbest_f)

    # -------------------------------
    def run(self):
        for t in range(self.iters):
            self.step(t)

    # -------------------------------
    def optimize(self):
        self.run()
        return self.history_best

    # -------------------------------
    def best_solution(self):
        return self.gbest_x, self.gbest_f, self.gbest_violation
