import math


def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    val = (
        math.exp(-(((x - mean) / std_dev) ** 2) / 2) / (2 * math.pi * std_dev**2) ** 0.5
    )
    return round(val, 5)


if __name__ == "__main__":
    assert normal_pdf(x=16, mean=15, std_dev=2.04) == 0.17342
