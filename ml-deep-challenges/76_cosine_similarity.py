import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape.")
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        raise ValueError("Vectors must be non-zero.")
    return round(v1 @ v2.T / (np.linalg.norm(v1) * np.linalg.norm(v2)), 3)


if __name__ == "__main__":
    assert cosine_similarity(v1=np.array([1, 2, 3]), v2=np.array([2, 4, 6])) == 1.0
