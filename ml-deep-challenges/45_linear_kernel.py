import numpy as np


def kernel_function(x1, x2):
    return np.dot(x1, x2)


def kernel_function_from_scratch(
    x1: list[int | float], x2: list[int | float]
) -> int | float:
    return sum(i * j for i, j in zip(x1, x2))


if __name__ == "__main__":
    assert kernel_function([1, 2, 3], [4, 5, 6]) == 32
    assert kernel_function_from_scratch([1, 2, 3], [4, 5, 6]) == 32
