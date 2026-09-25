"""Estimate a camera model when the focal length is unknown.

Used for historical photographs where we have no checkerboard calibration.
We know a handful of physical 3D landmarks and their pixel clicks in one
photo; we jointly solve for the focal length and camera pose so that the
landmarks reproject onto their clicked pixel positions.
"""

import numpy as np
from scipy.optimize import least_squares
from reconstruction.camera.projection import project_points


def fit_camera_unknown_focal(points_3d, points_2d, image_size,
                             focal_guess=None, principal_point=None):
    """Joint least-squares fit for (focal length, rvec, tvec).

    Parameters
    ----------
    points_3d, points_2d : (N, 3) and (N, 2) array-like
        Corresponding physical points and their clicked pixel positions.
    image_size : (width, height)
        Used to pick a sensible default focal-length guess (roughly the
        longer image dimension in pixels) and to default the principal
        point to the image centre.
    focal_guess : float, optional
        Initial focal length in pixels. If omitted, max(width, height).
    principal_point : (cx, cy), optional
        Assumed principal point. Defaults to the image centre — a
        reasonable prior when no calibration is available.

    Returns
    -------
    K, dist, rvec, tvec, rms, result
        Where ``dist`` is a zero vector (we do not fit distortion — too
        few samples), ``rms`` is the RMS reprojection error in pixels,
        and ``result`` is the raw SciPy OptimizeResult for diagnostics.
    """
    points_3d = np.asarray(points_3d, float)
    points_2d = np.asarray(points_2d, float)
    width, height = image_size
    if focal_guess is None:
        # Rough prior: image diagonal is comparable to focal length in
        # pixels for a "normal" field of view.
        focal_guess = max(width, height)
    cx, cy = principal_point or (width / 2.0, height / 2.0)

    K = np.array([[1., 0., cx], [0., 1., cy], [0., 0., 1.]])

    # Scene scale is used to pick a starting translation in front of the
    # scene: an initial guess of ~2x the point-cloud extent keeps the
    # optimizer away from the degenerate camera-inside-scene case.
    scale = max(float(np.linalg.norm(np.ptp(points_3d, axis=0))), 1.0)

    # x = [ log(f), rx, ry, rz, tx, ty, tz ]
    # Parameterising the focal length as log(f) keeps it strictly
    # positive without needing bounds, and makes the optimizer take
    # multiplicative steps in focal length (which is what we want, since
    # a factor-of-two change in focal length matters roughly equally at
    # any magnitude).
    x0 = np.r_[np.log(focal_guess), 0., 0., 0., 0., 0., max(2 * scale, 1.)]

    def residual(x):
        f = np.exp(x[0])
        K[0, 0] = K[1, 1] = f
        projected = project_points(points_3d, K, x[1:4], x[4:7])
        return (projected - points_2d).ravel()

    # soft_l1 loss down-weights the occasional clicked-badly outlier so
    # one poorly localized landmark doesn't dominate the fit.
    result = least_squares(residual, x0, method="trf",
                           loss="soft_l1", f_scale=3.0, max_nfev=5000)

    K[0, 0] = K[1, 1] = np.exp(result.x[0])
    rvec, tvec = result.x[1:4], result.x[4:7]
    errors = np.linalg.norm(
        project_points(points_3d, K, rvec, tvec) - points_2d, axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    return K, np.zeros(5), rvec, tvec, rms, result
