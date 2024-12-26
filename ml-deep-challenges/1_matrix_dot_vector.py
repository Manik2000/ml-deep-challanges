def matrix_dot_vector(
    a: list[list[int | float]], b: list[int | float]
) -> list[int | float]:
    n, m = len(a), len(b)
    if not all(len(a[i]) == m for i in range(n)):
        return -1
    return [sum(a[i][j] * b[j] for j in range(m)) for i in range(n)]


if __name__ == "__main__":
    assert matrix_dot_vector([[1, 2], [2, 4]], [1, 2]) == [5, 10]
    assert matrix_dot_vector([[1, 2], [2, 4], [6, 8], [12, 4]], [1, 2, 3]) == -1
    assert matrix_dot_vector([[1, 2, 3], [2, 4, 5], [6, 8, 9]], [1, 2, 3]) == [
        14,
        25,
        49,
    ]
