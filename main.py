from typing import List
from src.dataset import Dataset
from src.algorithms.algorithm import Algorithm
from src.algorithms.pso_algorithm import PsoAlgorithm
from src.algorithms.ga_algorithm import GaAlgorithm
import pandas as pd

RUN_TIMES = 10

dataset = Dataset.from_txts('instances/1/x_data.txt', 'instances/1/y_data.txt')

algorithms: List[Algorithm] = [
    PsoAlgorithm(dataset),
    GaAlgorithm(dataset),
]

dataframe_rows = []

dataframe = pd.DataFrame(columns = [
    'algorithm',
    'mean_fitness',
    'min_fitness',
])

for algorithm in algorithms:
    algorithm.run_many_times(RUN_TIMES)

    algorithm.plot_curve_graphic()
    algorithm.plot_best_fitness_graphic()

    dataframe_row = algorithm.get_dataframe_row()
    dataframe_rows.append(dataframe_row)

dataframe = pd.DataFrame(dataframe_rows, columns = ['algorithm', 'mean_fitness', 'min_fitness'])

print(dataframe)
