import math


def softmax(scores: list[float]) -> list[float]:
    denominator = sum([math.exp(score) for score in scores])
    return [round(math.exp(score) / denominator, 4) for score in scores]


if __name__ == "__main__":
    assert softmax([1, 2, 3]) == [0.09, 0.2447, 0.6652]
    assert softmax([1, 1, 1]) == [0.3333, 0.3333, 0.3333]
