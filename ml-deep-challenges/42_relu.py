def relu(z: float) -> float:
    return max(0, z)


if __name__ == "__main__":
    assert relu(0) == 0
    assert relu(1) == 1
    assert relu(-1) == 0
