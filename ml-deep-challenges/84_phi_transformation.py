from utils import compare_two_arrays


def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.
    """
    return [[i**j for j in range(degree + 1)] for i in data]


if __name__ == "__main__":
    result = phi_transform([1.0, 2.0], 2)
    assert compare_two_arrays(result[0], [1.0, 1.0, 1.0])
    assert compare_two_arrays(result[1], [1.0, 2.0, 4.0])
