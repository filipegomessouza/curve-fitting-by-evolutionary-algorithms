from abc import ABC, abstractmethod
from typing import Dict, List,Optional, Tuple
from src.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np

class Algorithm(ABC):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.individuals: List[np.ndarray] = []
        self.fitness_values: List[float] = []

        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = float('inf')

    def run_many_times(self, times: int):
        self.individuals = []
        self.fitness_values = []

        for _ in range(times):
            gbest_x, gbest_y = self.run()

            self.individuals.append(gbest_x)
            self.fitness_values.append(gbest_y)

        best_idx = np.argmin(self.fitness_values)
        self.best_individual = self.individuals[best_idx]
        self.best_fitness = self.fitness_values[best_idx]

    @abstractmethod
    def run(self) -> Tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def fitness(self, individual: np.ndarray) -> float:
        a: float = individual[0]
        b: float = individual[1]
        c: float = individual[2]

        x = self.dataset.x
        y = self.dataset.y

        y_pred = a + b * x + c * x * x

        return ((y - y_pred) ** 2).sum()

    def plot_curve_graphic(self):
        a: float = self.best_individual[0]
        b: float = self.best_individual[1]
        c: float = self.best_individual[2]

        x_plot = np.linspace(self.dataset.x.min(), self.dataset.x.max(), 300)
        y_plot = a + b * x_plot + c * x_plot * x_plot

        plt.figure(figsize=(10, 6))
        plt.scatter(self.dataset.x, self.dataset.y, color = 'blue', label = 'Data Points')
        plt.plot(x_plot, y_plot, color = 'red', label = 'Fitted Curve')
        plt.title(f'Curve Fitting using {self.name()}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()
        plt.savefig(f'graphics/curve_fit_{self.name().replace(" ", "_").lower()}.png', format='png')

    def get_dataframe_row(self) -> Dict[str, float]:
        return {
            'algorithm': self.name(),
            'mean_fitness': np.mean(self.fitness_values),
            'min_fitness': np.min(self.fitness_values)
        }
