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


class PSOAgent:
    def __init__(self, dim, bounds, obj_fn):
        self.x = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        self.v = [0.0 for _ in range(dim)]

        self.f, self.violation = obj_fn(self.x)

        self.pbest_x = self.x[:]
        self.pbest_f = self.f
        self.pbest_violation = self.violation


class PSO:
    def __init__(self, dim, pop_size, iters, bounds, obj_fn):
        self.dim = dim
        self.pop_size = pop_size
        self.iters = iters
        self.bounds = bounds
        self.obj_fn = obj_fn

        self.agents = [PSOAgent(dim, bounds, obj_fn) for _ in range(pop_size)]

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
                      agent.pbest_f, agent.pbest_violation):
                agent.pbest_x = agent.x[:]
                agent.pbest_f = agent.f
                agent.pbest_violation = agent.violation

            if better(agent.f, agent.violation,
                      self.gbest_f, self.gbest_violation):
                self.gbest_x = agent.x[:]
                self.gbest_f = agent.f
                self.gbest_violation = agent.violation

    # -------------------------------
    def optimize(self):
        self.run()
        return self.history_best

    # -------------------------------
    def update_velocities(self, t):
        w = 0.9 - 0.5 * (t / self.iters)
        c = 1.5
        c2 = 1.5

        for agent in self.agents:
            for d in range(self.dim):
                r1 = random.random()
                r2 = random.random()

                cognitive = c * r1 * (agent.pbest_x[d] - agent.x[d])
                social = c2 * r2 * (self.gbest_x[d] - agent.x[d])

                agent.v[d] = w * agent.v[d] + cognitive + social

    # -------------------------------
    def update_positions(self):
        for agent in self.agents:
            for d in range(self.dim):
                agent.x[d] += agent.v[d]

                low, high = self.bounds[d]
                agent.x[d] = max(min(agent.x[d], high), low)

    # -------------------------------
    def step(self, t):
        self.evaluate()
        self.update_velocities(t)
        self.update_positions()

        self.history_positions.append([agent.x[:] for agent in self.agents])
        self.history_best.append(self.gbest_f)

    # -------------------------------
    def run(self):
        for t in range(self.iters):
            self.step(t)

    # -------------------------------
    def best_solution(self):
        return self.gbest_x, self.gbest_f, self.gbest_violation
