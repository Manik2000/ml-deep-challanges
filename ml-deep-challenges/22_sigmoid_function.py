import math


def sigmoid(z: float) -> float:
    return round(1 / (1 + math.exp(-z)), 4)


if __name__ == "__main__":
    assert sigmoid(0) == 0.5
    assert sigmoid(1) == 0.7311
