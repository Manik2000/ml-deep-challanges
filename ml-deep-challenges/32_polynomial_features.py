from itertools import combinations_with_replacement

import numpy as np


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    n_samples, n_features = X.shape
    comb = combinations_with_replacement(range(n_features), degree)
    n_output_features = int(
        np.prod([n_features + degree - 1, degree]) / np.math.factorial(degree)
    )
    X_poly = np.empty((n_samples, n_output_features))
    for i, indices in enumerate(comb):
        X_poly[:, i] = np.prod(X[:, indices], axis=1)
    return X_poly
