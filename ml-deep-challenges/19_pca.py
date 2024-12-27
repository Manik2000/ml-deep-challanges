import numpy as np


def pca(data: np.ndarray, k: int) -> np.ndarray:
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, indices[:k]]
    return np.round(principal_components, 4)


if __name__ == "__main__":
    assert np.all(
        np.isclose(
            pca(np.array([[4, 2, 1], [5, 6, 7], [9, 12, 1], [4, 6, 7]]), 2),
            np.array([[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]),
        )
    )
    assert np.all(
        np.isclose(
            pca(np.array([[1, 2], [3, 4], [5, 6]]), k=1), np.array([[0.7071], [0.7071]])
        )
    )
