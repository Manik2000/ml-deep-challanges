import numpy as np


def calculate_contrast(img: np.ndarray) -> int:
    """
    Calculate the contrast of a grayscale image.
    Args:
            img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
    """
    return np.max(img) - np.min(img)


if __name__ == "__main__":
    assert calculate_contrast(np.array([[0, 50], [200, 255]])) == 255
