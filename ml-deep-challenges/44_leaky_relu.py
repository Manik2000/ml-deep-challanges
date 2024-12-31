def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    return z if z > 0 else alpha * z


if __name__ == "__main__":
    assert leaky_relu(0) == 0
    assert leaky_relu(1) == 1
    assert leaky_relu(-1) == -0.01
    assert leaky_relu(-2, alpha=0.1) == -0.2
