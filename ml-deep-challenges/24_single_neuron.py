import math


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> tuple[list[float], float]:
    predictions = [
        sigmoid(sum(feature[i] * weights[i] for i in range(len(feature))) + bias)
        for feature in features
    ]
    loss = sum((predictions[i] - labels[i]) ** 2 for i in range(len(labels))) / len(
        labels
    )
    return [round(prediction, 4) for prediction in predictions], round(loss, 4)


def assert_single_neuron_model(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    bias: float,
    expected: tuple[list[float], float],
):
    predictions, loss = single_neuron_model(features, labels, weights, bias)
    assert all(
        abs(i - j) < 1e-6 for i, j in zip(predictions, expected[0])
    ), f"{predictions} != {expected[0]}"
    assert loss == expected[1], f"{loss} != {expected[1]}"


if __name__ == "__main__":
    assert_single_neuron_model(
        [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]],
        [0, 1, 0],
        [0.7, -0.4],
        -0.1,
        ([0.4626, 0.4134, 0.6682], 0.3349),
    )
    assert_single_neuron_model(
        [[1, 2], [2, 3], [3, 1]],
        [1, 0, 1],
        [0.5, -0.2],
        0,
        ([0.525, 0.5987, 0.7858], 0.21),
    )
