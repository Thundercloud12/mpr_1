import math
import random


class GMOAgent:
    def __init__(self, dim, bounds, obj_fn):
        self.dim = dim
        self.bounds = bounds
        self.obj_fn = obj_fn
        
        
        self.x = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
        self.v = [0.0 for _ in range(dim)]
        
        
        self.x_best = self.x[:]
        self.f_best = float('inf')
        self.violation_best = float('inf')
        
        self.evaluate()
    
    def evaluate(self):
        f, violation = self.obj_fn(self.x)

        # Deb's rules
        if self.violation_best == 0 and violation == 0:
            # Both feasible → minimize objective
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x[:]

        elif violation == 0 and self.violation_best > 0:
            # New feasible beats old infeasible
            self.f_best = f
            self.violation_best = 0
            self.x_best = self.x[:]

        elif violation > 0 and self.violation_best > 0:
            # Both infeasible → minimize violation
            if violation < self.violation_best:
                self.f_best = f
                self.violation_best = violation
                self.x_best = self.x[:]

        return f, violation

    
    def update_velocity(self, v_new):
        self.v = v_new[:]
    
    def update_position(self, x_new):
        for d in range(self.dim):
            self.x[d] = max(min(x_new[d], self.bounds[d][1]), self.bounds[d][0])


class GMO:
    
    def __init__(self, dim, pop_size, iters, bounds, obj_fn):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iters = iters
        self.bounds = bounds
        self.obj_fn = obj_fn
        

        self.agents = [GMOAgent(dim, bounds, obj_fn) for _ in range(pop_size)]
        
        self.history_best = []
        self.history_positions = []
        self.current_iteration = 0

    def step(self):
        if self.current_iteration >= self.max_iters:
            return
        
        w_control = 1.0 - (self.current_iteration / self.max_iters)
        

        mu_t, sigma_t = self.compute_statistics()
        MF = self.compute_membership_functions(mu_t, sigma_t)
        DFI = self.compute_dual_fitness_index(MF)
        

        elites = self.select_elite_agents(DFI, self.current_iteration)
        guides = self.generate_guides(DFI, elites)

        std_dims, std_max = self.compute_dimension_std()
        mutated_guides = self.mutate_guides(guides, std_dims, std_max, w_control)
        

        self.update_positions(mutated_guides, w_control)
        self.evaluate_population()
        self.record_history()
        
        self.current_iteration += 1

    def compute_statistics(self):
     
        fitness_values = [
            agent.f_best if agent.violation_best == 0
            else agent.f_best + 1e6 * agent.violation_best
            for agent in self.agents
        ]

        
        mu_t = sum(fitness_values) / self.pop_size
        variance = sum((f - mu_t)**2 for f in fitness_values) / self.pop_size
        sigma_t = math.sqrt(variance) + 1e-10  # Avoid division by zero
        
        return mu_t, sigma_t
    

    def compute_membership_functions(self, mu_t, sigma_t):
  
        a_param = -4 / (sigma_t * math.sqrt(math.e))
        
        MF = []
        for agent in self.agents:
            mf_value = 1.0 / (1.0 + math.exp(-a_param * (agent.f_best - mu_t)))
            MF.append(mf_value)
        
        return MF
    

    def compute_dual_fitness_index(self, MF):

        total_prod = 1.0
        for m in MF:
            total_prod *= m
        
        DFI = []
        for m in MF:
            dfi_value = total_prod / (m + 1e-10)  
            DFI.append(dfi_value)
        
        return DFI
    

    def select_elite_agents(self, DFI, iteration):

        progress = iteration / self.max_iters
        n_best_count = int(2 + (self.pop_size - 2) * (1 - progress))

        n_best_count = max(n_best_count, 2)  
        
        sorted_indices = sorted(
            range(self.pop_size),
            key=lambda i: (
                self.agents[i].violation_best,
                -DFI[i]
            )
        )

        
        elites = sorted_indices[:n_best_count]
        return elites
    

    def generate_guides(self, DFI, elites):

        guides = []
        
        for i in range(self.pop_size):
            sum_weighted_pos = [0.0] * self.dim
            sum_weights = 0.0

            for elite_idx in elites:
                if elite_idx == i:
                    continue  

                if self.agents[elite_idx].violation_best > 0:
                    continue
                
                weight = DFI[elite_idx]
                sum_weights += weight

                for d in range(self.dim):
                    sum_weighted_pos[d] += weight * self.agents[elite_idx].x_best[d]
            
            # Normalize by total weight
            guide = [
                pos / (sum_weights + 1e-10) for pos in sum_weighted_pos
            ]
            guides.append(guide)
        
        return guides

    def compute_dimension_std(self):

        std_dims = []
        
        for d in range(self.dim):

            col = [self.agents[i].x_best[d] for i in range(self.pop_size)]
            
            mean_col = sum(col) / self.pop_size
            var_col = sum((x - mean_col)**2 for x in col) / self.pop_size
            std_col = math.sqrt(var_col)
            
            std_dims.append(std_col)
        
        std_max = max(std_dims) if std_dims else 1e-10
        return std_dims, std_max

    def mutate_guides(self, guides, std_dims, std_max, w_control):
   
        mutated_guides = []
        
        for guide in guides:
            mutated_guide = []
            for d in range(self.dim):
                mutation = w_control * random.gauss(0, 1) * (std_max - std_dims[d])
                mutated_value = guide[d] + mutation
                mutated_guide.append(mutated_value)
            
            mutated_guides.append(mutated_guide)
        
        return mutated_guides

    def update_positions(self, mutated_guides, w_control):

        for i in range(self.pop_size):

            phi = 1.0 + (2.0 * random.random() - 1.0) * w_control
            
            new_velocity = []
            new_position = []
            
            for d in range(self.dim):

                v_new = (w_control * self.agents[i].v[d] +
                         phi * (mutated_guides[i][d] - self.agents[i].x[d]))
                
                new_velocity.append(v_new)

                x_new = self.agents[i].x[d] + v_new
                new_position.append(x_new)
            
            # Apply updates
            self.agents[i].update_velocity(new_velocity)
            self.agents[i].update_position(new_position)
    

    def evaluate_population(self):

        for agent in self.agents:
            agent.evaluate()
    

    def record_history(self):

        best_fitness = min(agent.f_best for agent in self.agents)
        self.history_best.append(best_fitness)
        
        positions = [agent.x[:] for agent in self.agents]
        self.history_positions.append(positions)
    

    def optimize(self):
        self.current_iteration = 0
        for _ in range(self.max_iters):
            self.step()
    

    def best_solution(self):
        best_agent = min(
            self.agents,
            key=lambda a: (a.violation_best, a.f_best)
        )
        
        return best_agent.x_best, best_agent.f_best, best_agent.violation_best
    
    def run(self):
        self.optimize()