import cv2
import numpy as np

from reconstruction.image.clicking import fit_ellipse_near_hint


def test_fit_ellipse_near_rough_circle_hint():
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.circle(image, (220, 170), 42, (255, 255, 255), 2, cv2.LINE_AA)

    center, axes, _ = fit_ellipse_near_hint(image, (224, 167), 39)

    assert np.linalg.norm(center - (220, 170)) < 2
    assert np.allclose(axes, (84, 84), atol=5)