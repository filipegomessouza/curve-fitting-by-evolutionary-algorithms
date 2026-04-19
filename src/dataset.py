from typing import List

class Dataset:
    def __init__(self, x_data: List[float], y_data: List[float]):
        self.__validate_data(x_data, y_data)
        self.x_data = x_data
        self.y_data = y_data

    def __validate_data(self, x_data: List[float], y_data: List[float]):
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length.")

        if not all(isinstance(x, (int, float)) for x in x_data):
            raise ValueError("All elements in x_data must be numbers.")

        if not all(isinstance(y, (int, float)) for y in y_data):
            raise ValueError("All elements in y_data must be numbers.")

    @classmethod
    def from_txts(cls, x_data_path: str, y_data_path: str):
        x_data = cls.__read_txt(x_data_path)
        y_data = cls.__read_txt(y_data_path)

        return cls(x_data, y_data)

    @classmethod
    def __read_txt(cls, file_path: str) -> List[float]:
        with open(file_path, 'r') as file:
            return [float(line.strip()) for line in file.readlines()]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx: int):
        return self.x_data[idx], self.y_data[idx]

    def x(self, idx: int):
        return self.x_data[idx]

    def y(self, idx: int):
        return self.y_data[idx]
