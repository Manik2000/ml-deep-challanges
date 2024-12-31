import math


def poisson_probability(k, lam):
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    """
    val = (lam**k) * math.exp(-lam) / math.factorial(k)
    return round(val, 5)


if __name__ == "__main__":
    assert poisson_probability(k=3, lam=5) == 0.14037
