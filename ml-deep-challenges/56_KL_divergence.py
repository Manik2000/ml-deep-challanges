import numpy as np


def kl_divergence_normal(
    mu_p: float, sigma_p: float, mu_q: float, sigma_q: float
) -> float:
    sq_sigmas_ratio = (sigma_p / sigma_q) ** 2
    return (
        sq_sigmas_ratio + ((mu_p - mu_q) / sigma_q) ** 2 - np.log(sq_sigmas_ratio) - 1
    ) / 2


if __name__ == "__main__":
    assert kl_divergence_normal(0.0, 1.0, 0.0, 1.0) == 0.0
    assert kl_divergence_normal(0.0, 1.0, 1.0, 1.0) == 0.5
