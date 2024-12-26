import numpy as np


def reshape_matrix(
    a: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
    return np.array(a).reshape(new_shape).tolist()


if __name__ == "__main__":
    assert reshape_matrix([[1, 2, 3, 4], [5, 6, 7, 8]], (4, 2)) == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]
    assert reshape_matrix([[1, 2, 3], [4, 5, 6]], (3, 2)) == [[1, 2], [3, 4], [5, 6]]
