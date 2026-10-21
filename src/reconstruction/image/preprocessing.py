"""Basic image I/O.

All pixel coordinates in this project are expressed in the camera's *sensor*
frame: the pixel grid exactly as stored in the file, before any EXIF
``Orientation`` rotation. ``cv2.imread`` applies EXIF rotation by default,
which silently swaps width/height and rotates the (u, v) axes of portrait
shots relative to the intrinsics from :mod:`reconstruction.camera.calibration`.
Always load photographs through :func:`load_image` so every view shares the
same coordinate frame as the calibration.
"""

import cv2

_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def load_image(path):
    """Load an image as BGR in the sensor frame (EXIF orientation ignored)."""
    image = cv2.imread(str(path), _FLAGS)
    if image is None:
        raise FileNotFoundError(path)
    return image


def image_size(image):
    """Return ``(width, height)`` of an OpenCV image array."""
    return image.shape[1], image.shape[0]
