import random

from utils import time_func


def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    if len(a[0]) != len(b):
        return -1
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def matrixmul_2(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    if len(a[0]) != len(b):
        return -1

    vals = []
    for i in range(len(a)):
        hold = []
        for j in range(len(b[0])):
            val = 0
            for k in range(len(b)):
                val += a[i][k] * b[k][j]

            hold.append(val)
        vals.append(hold)

    return vals


@time_func
def matrixmul_check(
    func: callable, a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    return func(a, b)


if __name__ == "__main__":
    assert matrixmul(
        [[1, 2, 3], [2, 3, 4], [5, 6, 7]], [[3, 2, 1], [4, 3, 2], [5, 4, 3]]
    ) == [[26, 20, 14], [38, 29, 20], [74, 56, 38]]
    assert matrixmul([[0, 0], [2, 4], [1, 2]], [[0, 0], [2, 4]]) == [
        [0, 0],
        [8, 16],
        [4, 8],
    ]

    benchmark_matrix = [
        [random.randint(0, 100) for _ in range(1000)] for _ in range(1000)
    ]

    matrixmul_check(matrixmul, benchmark_matrix, benchmark_matrix)
    matrixmul_check(matrixmul_2, benchmark_matrix, benchmark_matrix)
