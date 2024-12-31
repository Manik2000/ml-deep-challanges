import numpy as np


def rmse(y_true, y_pred):
    shape = y_true.shape
    n_elems = shape[0] * shape[1] if len(shape) > 1 else shape[0]
    rmse_res = np.sqrt(((y_true - y_pred) ** 2).sum() / n_elems)
    return round(rmse_res, 3)


if __name__ == "__main__":
    assert rmse(np.array([3, -0.5, 2, 7]), np.array([2.5, 0.0, 2, 8])) == 0.612
