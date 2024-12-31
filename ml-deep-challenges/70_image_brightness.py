def calculate_brightness(img: list[list[float]]):
    if not img or any(not row for row in img):
        return -1
    n_cols = len(img[0])
    if not all(len(row) == n_cols for row in img[1:]):
        return -1
    if not all(0 <= pixel <= 255 for row in img for pixel in row):
        return -1
    return round(sum(sum(row) for row in img) / (len(img) * n_cols), 2)


if __name__ == "__main__":
    assert calculate_brightness([[100, 200], [50, 150]]) == 125.0
