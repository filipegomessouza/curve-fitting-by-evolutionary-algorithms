from src.algorithms.algorithm import Algorithm
from sko.PSO import PSO
import numpy as np

class PsoAlgorithm(Algorithm):
    RUNS = 10
    POPULATION_SIZE = 100
    MAX_ITERATIONS = 1000
    LOWER_BOUND = -5
    UPPER_BOUND = 5
    W = 0.5
    C1 = 0.5
    C2 = 0.5

    def name(self) -> str:
        return 'Particle Swarm Optimization'

    def run(self):
        self.individuals = []
        self.fitness_values = []

        for _ in range(self.RUNS):
            gbest_x, gbest_y = self.run_pso()

            self.individuals.append(gbest_x)
            self.fitness_values.append(gbest_y)

        best_idx = np.argmin(self.fitness_values)
        self.best_individual = self.individuals[best_idx]
        self.best_fitness = self.fitness_values[best_idx]

    def run_pso(self):
        pso = PSO(
            func = self.fitness,
            n_dim = 3,
            pop = self.POPULATION_SIZE,
            max_iter = self.MAX_ITERATIONS,
            lb = [self.LOWER_BOUND, self.LOWER_BOUND, self.LOWER_BOUND],
            ub = [self.UPPER_BOUND, self.UPPER_BOUND, self.UPPER_BOUND],
            w = self.W,
            c1 = self.C1,
            c2 = self.C2,
        )

        pso.run()

        return pso.gbest_x, pso.gbest_y[0]
