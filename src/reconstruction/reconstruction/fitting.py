import numpy as np
from scipy.optimize import least_squares
from reconstruction.camera.projection import project_points

def fit_camera_unknown_focal(points_3d, points_2d, image_size,
                             focal_guess=None, principal_point=None):
    points_3d = np.asarray(points_3d, float)
    points_2d = np.asarray(points_2d, float)
    width, height = image_size
    if focal_guess is None:
        focal_guess = max(width, height)
    cx, cy = principal_point or (width/2.0, height/2.0)

    K = np.array([[1., 0., cx], [0., 1., cy], [0., 0., 1.]])
    scale = max(float(np.linalg.norm(np.ptp(points_3d, axis=0))), 1.0)
    x0 = np.r_[np.log(focal_guess), 0., 0., 0., 0., 0., max(2*scale, 1.)]

    def residual(x):
        f = np.exp(x[0])
        K[0, 0] = K[1, 1] = f
        projected = project_points(points_3d, K, x[1:4], x[4:7])
        return (projected - points_2d).ravel()

    result = least_squares(residual, x0, method="trf",
                           loss="soft_l1", f_scale=3.0, max_nfev=5000)
    K[0, 0] = K[1, 1] = np.exp(result.x[0])
    rvec, tvec = result.x[1:4], result.x[4:7]
    errors = np.linalg.norm(project_points(points_3d, K, rvec, tvec) - points_2d, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    return K, np.zeros(5), rvec, tvec, rms, result
