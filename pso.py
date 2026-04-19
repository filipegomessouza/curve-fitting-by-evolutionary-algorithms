from typing import Tuple
from src.dataset import Dataset
import numpy as np
from sko.PSO import PSO
import pandas as pd
import matplotlib.pyplot as plt

dataset = Dataset.from_txts('instances/1/x_data.txt', 'instances/1/y_data.txt')

def f(a: float, b: float, c: float, x: np.ndarray) -> np.ndarray:
    return a + b * x + c * x * x

def fitness(x: np.ndarray) -> float:
    a, b, c = x

    return np.sum((dataset.y - f(a, b, c, dataset.x)) ** 2)

def run_pso() -> Tuple[np.ndarray, float]:
    pso = PSO(
        func = fitness,
        n_dim = 3,
        pop = 100,
        max_iter = 1000,
        lb = [-5, -5, -5],
        ub = [5, 5, 5],
        w = 0.5,
        c1 = 0.5,
        c2 = 0.5,
    )

    pso.run()

    return pso.gbest_x, pso.gbest_y[0]

results_x = []
results_y = []

for _ in range(10):
    gbest_x, gbest_y = run_pso()
    results_x.append(gbest_x)
    results_y.append(gbest_y)

    # print(f"Best parameters: {gbest_x}, Best fitness: {gbest_y}")

results_x = np.array(results_x)
results_y = np.array(results_y)

df = pd.DataFrame({
    'mean_fitness': [np.mean(results_y)],
    'min_fitness': [np.min(results_y)]
})

# Encontrar os melhores parâmetros
best_idx = np.argmin(results_y)
best_a, best_b, best_c = results_x[best_idx]

# Criar gráfico
x_plot = np.linspace(dataset.x.min(), dataset.x.max(), 300)
y_plot = f(best_a, best_b, best_c, x_plot)

plt.figure(figsize=(10, 6))
plt.scatter(dataset.x, dataset.y, label='Dados', color='blue', s=50)
plt.plot(x_plot, y_plot, label=f'Parábola ajustada (a={best_a:.4f}, b={best_b:.4f}, c={best_c:.4f})', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de Parábola usando PSO')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curve_fit_pso.png', format = 'png')
