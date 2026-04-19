from typing import Tuple
from src.algorithms.algorithm import Algorithm
from sko.GA import GA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GaAlgorithm(Algorithm):
    POPULATION_SIZE = 100
    MAX_ITERATIONS = 1000
    LOWER_BOUND = -5
    UPPER_BOUND = 5
    MUTATION_PROBABILITY = 0.01
    PRECISION = 1e-7

    def name(self) -> str:
        return 'Genetic Algorithm'

    def run(self) -> Tuple[np.ndarray, float]:
        ga = GA(
            func = self.fitness,
            n_dim = 3,
            size_pop = self.POPULATION_SIZE,
            max_iter = self.MAX_ITERATIONS,
            prob_mut = self.MUTATION_PROBABILITY,
            lb = [self.LOWER_BOUND, self.LOWER_BOUND, self.LOWER_BOUND],
            ub = [self.UPPER_BOUND, self.UPPER_BOUND, self.UPPER_BOUND],
            precision = self.PRECISION,
        )

        ga.run()

        fitness_history = pd.DataFrame(ga.all_history_Y).min(axis=1).tolist()
        self.fitness_values_history.append(fitness_history)

        return ga.best_x, ga.best_y
