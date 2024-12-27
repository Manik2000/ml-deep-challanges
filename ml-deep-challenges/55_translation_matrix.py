import numpy as np

Number = int | float


def translate_object(points: list[list[Number]], tx: Number, ty: Number):
    points = np.hstack([np.array(points), np.ones((len(points), 1))])
    translation_matrix = np.array([[1, 0, 0], [0, 1, 0], [tx, ty, 1]])
    translated_points = points @ translation_matrix
    return translated_points[:, :-1].tolist()


if __name__ == "__main__":
    assert np.all(
        np.isclose(
            translate_object([[0, 0], [1, 0], [0.5, 1]], 2, 3),
            [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]],
        )
    )
    assert np.all(
        np.isclose(
            translate_object([[0, 0], [1, 0], [1, 1], [0, 1]], -1, 2),
            [[-1.0, 2.0], [0.0, 2.0], [0.0, 3.0], [-1.0, 3.0]],
        )
    )
