def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    sq_discriminant = (trace**2 - 4 * determinant) ** 0.5
    lambda_1 = (trace + sq_discriminant) / 2
    lambda_2 = (trace - sq_discriminant) / 2
    return [lambda_1, lambda_2]


if __name__ == "__main__":
    assert calculate_eigenvalues([[2, 1], [1, 2]]) == [3.0, 1.0]
    assert calculate_eigenvalues([[4, -2], [1, 1]]) == [3.0, 2.0]
