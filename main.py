from src.dataset import Dataset

dataset = Dataset.from_txts('instances/1/x_data.txt', 'instances/1/y_data.txt')

print(f"Dataset size: {len(dataset)}")

print(dataset.x(1), dataset.y(1))
