"""Intrinsic camera calibration from checkerboard photographs.

The user photographs a printed checkerboard from many orientations, keeping
focal length, focus, resolution, cropping and image processing fixed. The
same optical configuration must then be used for reconstruction photographs
(see docs/camera_calibration.md). The output is the intrinsic matrix K and
the lens distortion coefficients; extrinsics for the calibration board are
discarded because they carry no meaning in the reconstruction coordinate
system.
"""

import cv2
import numpy as np


def calibrate_checkerboard(image_paths, pattern_size=(9, 6), square_size=25.0):
    """Estimate intrinsics + distortion from a set of checkerboard photos.

    Parameters
    ----------
    image_paths : iterable of path-like
        Photographs of the same checkerboard in different poses.
    pattern_size : (cols, rows)
        Number of *inner* corners per row and per column of the board.
    square_size : float
        Physical side length of one board square, in the units you want
        the camera translation to be reported in (typically millimetres).

    Returns
    -------
    Whatever ``cv2.calibrateCamera`` returns: ``(rms, K, dist, rvecs, tvecs)``.
    The rvecs / tvecs describe the board pose in each individual image and
    are usually discarded by callers.
    """
    # Build the canonical 3D object points for a planar board lying in Z=0,
    # with corners at integer multiples of ``square_size``. The ordering
    # (row-major) matches the order in which findChessboardCorners returns
    # detected corners, which is required for the pairing below.
    obj = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    obj *= square_size

    object_points, image_points = [], []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # OpenCV wants image_size as (width, height), i.e. (cols, rows).
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK)
        if not found:
            # Skip photos where the whole board is not visible / detectable.
            continue
        # Refine corners to sub-pixel accuracy — this materially improves
        # the calibration RMS.
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
        object_points.append(obj.copy())
        image_points.append(corners)

    if len(object_points) < 5:
        # Five is a soft lower bound; fewer views leave focal length and
        # distortion poorly constrained.
        raise RuntimeError("Fewer than five usable checkerboard images.")

    return cv2.calibrateCamera(object_points, image_points, image_size, None, None)
