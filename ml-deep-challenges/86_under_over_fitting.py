def model_fit_quality(training_accuracy: float, test_accuracy: float) -> int:
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: int, one of '1', '-1', or '0'.
    """
    if training_accuracy - test_accuracy > 0.2:
        return 1
    elif test_accuracy < 0.7 and training_accuracy < 0.7:
        return -1
    return 0


if __name__ == "__main__":
    assert model_fit_quality(training_accuracy=0.95, test_accuracy=0.65) == 1