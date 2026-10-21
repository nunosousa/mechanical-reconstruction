import cv2
import numpy as np
import pytest
from PIL import Image

from reconstruction.camera.calibration import calibrate_checkerboard
from reconstruction.camera.pose import check_image_size
from reconstruction.image.preprocessing import image_size, load_image


def _write_rotated_jpeg(path):
    """A landscape 60x40 JPEG flagged EXIF Orientation=8 (display as portrait)."""
    pixels = np.zeros((40, 60, 3), dtype=np.uint8)
    pixels[:, :30] = 255
    exif = Image.Exif()
    exif[0x0112] = 8
    Image.fromarray(pixels).save(path, exif=exif)


def test_load_image_ignores_exif_orientation(tmp_path):
    path = tmp_path / "portrait.jpg"
    _write_rotated_jpeg(path)

    assert cv2.imread(str(path)).shape[:2] == (60, 40)
    image = load_image(path)
    assert image_size(image) == (60, 40)
    assert image[20, 5].mean() > 200 and image[20, 55].mean() < 50


def test_check_image_size_rejects_rotated_frame():
    camera = {"image_size": (6000, 4000)}
    check_image_size(camera, (6000, 4000))
    with pytest.raises(ValueError, match="does not match"):
        check_image_size(camera, (4000, 6000))


def test_calibration_rejects_mixed_image_sizes(tmp_path):
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    cv2.imwrite(str(paths[0]), np.zeros((40, 60, 3), np.uint8))
    cv2.imwrite(str(paths[1]), np.zeros((60, 40, 3), np.uint8))

    with pytest.raises(ValueError, match="share one size"):
        calibrate_checkerboard(paths)
