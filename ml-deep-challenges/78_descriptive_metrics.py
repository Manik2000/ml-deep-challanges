import numpy as np


def descriptive_statistics(data):
    n = len(data)
    mean = np.sum(data) / n

    sorted_data = np.sort(data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    counts = np.bincount(data)
    mode = np.argmax(counts)

    variance = np.sum((data - mean) ** 2) / n
    std_dev = np.sqrt(variance)

    percentiles = np.percentile(data, [25, 50, 75])

    iqr = percentiles[2] - percentiles[0]

    stats_dict = {
        "mean": float(mean),
        "median": float(median),
        "mode": mode,
        "variance": np.round(variance, 4),
        "standard_deviation": np.round(std_dev, 4),
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr,
    }
    return stats_dict


if __name__ == "__main__":
    for i, j in zip(
        descriptive_statistics([10, 20, 30, 40, 50]).values(),
        {
            "mean": 30.0,
            "median": 30.0,
            "mode": 10,
            "variance": 200.0,
            "standard_deviation": 14.1421,
            "25th_percentile": 20.0,
            "50th_percentile": 30.0,
            "75th_percentile": 40.0,
            "interquartile_range": 20.0,
        }.values(),
    ):
        assert i == j, f"{i} != {j}"
