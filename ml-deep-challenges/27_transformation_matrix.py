import numpy as np
from utils import compare_two_arrays


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    return np.round(np.linalg.solve(np.array(C), np.array(B)), 4).tolist()


if __name__ == "__main__":
    assert compare_two_arrays(
        transform_basis(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]
        ),
        [
            [-0.6772, -0.0126, 0.2342],
            [-0.0184, 0.0505, -0.0275],
            [0.5732, -0.0345, -0.0569],
        ],
    )
