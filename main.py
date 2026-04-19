from src.dataset import Dataset
from src.algorithms.pso import PsoAlgorithm
import pandas as pd

dataset = Dataset.from_txts('instances/1/x_data.txt', 'instances/1/y_data.txt')

algorithms = [
    PsoAlgorithm(dataset),
]

dataframe_rows = []

dataframe = pd.DataFrame(columns = [
    'algorithm',
    'mean_fitness',
    'min_fitness',
])

for algorithm in algorithms:
    algorithm.run()
    algorithm.plot_curve_graphic()

    dataframe_row = algorithm.get_dataframe_row()
    dataframe_rows.append(dataframe_row)

dataframe = pd.DataFrame(dataframe_rows, columns = ['algorithm', 'mean_fitness', 'min_fitness'])

print(dataframe)
