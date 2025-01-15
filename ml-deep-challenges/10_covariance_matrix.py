import numpy as np
from utils import compare_two_arrays


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    np_vectors = np.array(vectors).T
    if np_vectors.size > 0:
        np_vectors = np_vectors - np.mean(np_vectors, axis=0)
        covariances = np.dot(np_vectors.T, np_vectors) / (np_vectors.shape[0] - 1)
    return covariances.tolist()


def covariance_from_scratch(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_observations = len(vectors[0])
    means = [sum(vector) / n_observations for vector in vectors]

    covariance = [[0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            covariance[i][j] = sum(
                (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                for k in range(n_observations)
            ) / (n_observations - 1)

    return covariance


if __name__ == "__main__":
    compare_two_arrays(
        covariance_from_scratch([[1, 2, 3], [4, 5, 6]]), [[1, 1], [1, 1]]
    )
