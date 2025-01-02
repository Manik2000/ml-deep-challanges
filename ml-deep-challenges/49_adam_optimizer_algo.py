import numpy as np
from utils import compare_two_arrays


def adam_optimizer(
    f,
    grad,
    x0,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10,
):
    m, v = np.zeros_like(x0), np.zeros_like(x0)
    x = x0
    for t in range(1, num_iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x


if __name__ == "__main__":

    def objective_function(x):
        return x[0] ** 2 + x[1] ** 2

    def gradient(x):
        return np.array([2 * x[0], 2 * x[1]])

    x_opt = adam_optimizer(objective_function, gradient, np.array([1.0, 1.0]))

    compare_two_arrays(x_opt, np.array([0.99000325, 0.99000325]))
