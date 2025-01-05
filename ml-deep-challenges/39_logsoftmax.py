import numpy as np
from utils import compare_two_arrays


def log_softmax(scores: list | np.ndarray) -> np.ndarray:
    scores = np.array(scores)
    x_max = np.max(scores)
    log_sum = np.log(np.sum(np.exp(scores - x_max)))
    return scores - x_max - log_sum


if __name__ == "__main__":
    assert compare_two_arrays(
        np.array([-2.40760596, -1.40760596, -0.40760596]), log_softmax([1, 2, 3])
    )
