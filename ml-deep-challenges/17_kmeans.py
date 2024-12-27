def calculate_centroid(points: list[tuple[float, ...]]) -> tuple[float, ...]:
    return tuple(sum(i) / len(points) for i in zip(*points))


def sq_euclidean_distance(p1: tuple[float, ...], p2: tuple[float, ...]) -> float:
    return sum((i - j) ** 2 for i, j in zip(p1, p2))


def get_centroid_idx(
    point: tuple[float, ...], centroids: list[tuple[float, ...]]
) -> int:
    return min(
        range(len(centroids)), key=lambda i: sq_euclidean_distance(point, centroids[i])
    )


def recalcualte_centroids(points: list[tuple[float, ...]], clusters_idx: list[int]):
    return [
        calculate_centroid(
            [points[i] for i in range(len(points)) if clusters_idx[i] == j]
        )
        for j in sorted(set(clusters_idx))
    ]


def round_tuple(t: tuple[float, ...], precision: int) -> tuple[float, ...]:
    return tuple(round(i, precision) for i in t)


def k_means_clustering(
    points: list[tuple[float, ...]],
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int,
) -> list[tuple[float, ...]]:
    cluster_idx = [get_centroid_idx(point, initial_centroids) for point in points]
    for _ in range(max_iterations):
        new_centroids = recalcualte_centroids(points, cluster_idx)
        new_cluster_idx = [get_centroid_idx(point, new_centroids) for point in points]
        cluster_idx = new_cluster_idx
    return [round_tuple(i, 4) for i in new_centroids]


if __name__ == "__main__":
    assert k_means_clustering(
        [(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)],
        [(1, 1, 1), (10, 10, 10)],
        10,
    ) == [(1.0, 1.0, 1.0), (10.3333, 10.6667, 10.3333)]
