from functools import wraps
from typing import Any, Callable, Tuple

import numpy as np


def time_func(f: Callable) -> Any:
    @wraps(f)
    def wrapper(*args: Tuple, **kwargs: dict) -> Any:
        from time import perf_counter

        start = perf_counter()
        result = f(*args, **kwargs)
        print(f"{f.__name__} took {perf_counter() - start} seconds")
        return result

    return wrapper


def compare_two_arrays(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    return np.all(np.isclose(arr1, arr2))
