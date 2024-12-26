import numpy as np


def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value, min_value = np.max(data, axis=0), np.min(data, axis=0)
    normalized_data = (data - min_value) / (max_value - min_value)
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return np.round(standardized_data, 4), np.round(normalized_data, 4)


def test_feature_scaling(
    test_input: np.ndarray, expected_output: tuple[np.ndarray, np.ndarray]
):
    standardized, normalized = feature_scaling(test_input)
    assert np.all(np.isclose(standardized, expected_output[0]))
    assert np.all(np.isclose(normalized, expected_output[1]))


if __name__ == "__main__":
    test_feature_scaling(
        np.array([[1, 2], [3, 4], [5, 6]]),
        (
            np.array([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]),
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        ),
    )
