import numpy as np


def calculate_dot_product(vec1, vec2) -> float:
    """
    Calculate the dot product of two vectors.
    Args:
            vec1 (numpy.ndarray): 1D array representing the first vector.
            vec2 (numpy.ndarray): 1D array representing the second vector.
    """
    return np.sum(vec1 * vec2)


if __name__ == "__main__":
    assert (
        calculate_dot_product(vec1=np.array([1, 2, 3]), vec2=np.array([4, 5, 6])) == 32
    )
