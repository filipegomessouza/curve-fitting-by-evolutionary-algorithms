from typing import Tuple
from src.algorithms.algorithm import Algorithm
from sko.PSO import PSO
import numpy as np

class PsoAlgorithm(Algorithm):
    POPULATION_SIZE = 100
    MAX_ITERATIONS = 1000
    LOWER_BOUND = -5
    UPPER_BOUND = 5
    W = 0.729
    C1 = 2
    C2 = 2

    def name(self) -> str:
        return 'Particle Swarm Optimization'

    def run(self) -> Tuple[np.ndarray, float]:
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
        self.best_fitness_values_history.append([fitness[0] for fitness in pso.gbest_y_hist])

        return pso.gbest_x, pso.gbest_y[0]
