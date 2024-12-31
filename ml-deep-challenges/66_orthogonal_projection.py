from utils import compare_two_arrays


def dot_product(v1: list[int | float], v2: list[int | float]) -> float:
    return sum([x * y for x, y in zip(v1, v2)])


def orthogonal_projection(
    v: list[int | float], L: list[int | float]
) -> list[int | float]:
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    return [dot_product(v, L) / dot_product(L, L) * x for x in L]


if __name__ == "__main__":
    assert compare_two_arrays(orthogonal_projection([3, 4], [1, 0]), [3, 0])
