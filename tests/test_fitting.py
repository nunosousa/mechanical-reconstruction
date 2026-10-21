import cv2
import numpy as np

from reconstruction.camera.projection import camera_matrix, project_points
from reconstruction.reconstruction.fitting import fit_camera_unknown_focal


def test_fit_unknown_focal_recovers_side_view():
    X = np.array([[72, 0, 0], [23, -26, 67], [23, 48, 76], [23, 104, 261],
                  [23, 145, 207], [44, 173, 264], [17, 210, 224]], float)
    # Camera looking along -X at the frame (side view), as in real photos.
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([-100., 130., 1500.])
    K = camera_matrix(8000, (6000, 4000))
    x = project_points(X, K, rvec, tvec)

    K_fit, _, _, _, rms, _ = fit_camera_unknown_focal(X, x, (6000, 4000))

    assert rms < 1e-3
    assert abs(K_fit[0, 0] - 8000) < 1
