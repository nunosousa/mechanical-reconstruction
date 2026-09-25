"""Building blocks for the pinhole projection model.

Everything here follows OpenCV conventions: +Z points forward from the
camera, image coordinates are (u, v) with the origin at the top-left and
u growing to the right, v downward.
"""

import cv2
import numpy as np


def camera_matrix(focal_length, image_size, principal_point=None):
    """Assemble a pinhole intrinsic matrix K.

    Uses a single focal length for both axes (square pixels) and defaults
    the principal point to the image centre — a reasonable prior when no
    calibration is available.
    """
    width, height = image_size
    cx, cy = principal_point or (width / 2.0, height / 2.0)
    return np.array([[focal_length, 0, cx],
                     [0, focal_length, cy],
                     [0, 0, 1]], dtype=float)


def project_points(points_3d, K, rvec, tvec, dist=None):
    """Project world-frame 3D points into pixel coordinates."""
    dist = np.zeros(5) if dist is None else dist
    # cv2.projectPoints wants shape (N, 1, 3) for the input.
    pts = np.asarray(points_3d, dtype=float).reshape(-1, 1, 3)
    out, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return out.reshape(-1, 2)


def projection_matrix(K, rvec, tvec):
    """Return the 3x4 projection matrix P = K [R | t].

    Handy for linear triangulation, where OpenCV wants the full P rather
    than the separated (K, rvec, tvec) form. Note that lens distortion is
    not embedded in P; if it matters, undistort the pixel coordinates
    first (this project's cameras have negligible distortion on the axes
    we care about, so we currently do not).
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    t = np.asarray(tvec, dtype=float).reshape(3, 1)
    return K @ np.hstack((R, t))
