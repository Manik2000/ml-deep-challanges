import math


def binomial_probability(n, k, p):
    """
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
    probability = p**k * (1 - p) ** (n - k) * math.comb(n, k)
    return round(probability, 5)


if __name__ == "__main__":
    assert (
        binomial_probability(n=6, k=2, p=0.5) == 0.23438
    ), f"Expected 0.23438 but got {binomial_probability(n=6, k=2, p=0.5)}"
