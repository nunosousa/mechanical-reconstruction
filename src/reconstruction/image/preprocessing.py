"""Basic image I/O."""

import cv2


def load_image(path):
    """Load an image as BGR (OpenCV's native ordering) or raise if missing."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image
