import cv2
import numpy as np

def camera_matrix(focal_length, image_size, principal_point=None):
    width, height = image_size
    cx, cy = principal_point or (width / 2.0, height / 2.0)
    return np.array([[focal_length, 0, cx],
                     [0, focal_length, cy],
                     [0, 0, 1]], dtype=float)

def project_points(points_3d, K, rvec, tvec, dist=None):
    dist = np.zeros(5) if dist is None else dist
    pts = np.asarray(points_3d, dtype=float).reshape(-1, 1, 3)
    out, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return out.reshape(-1, 2)

def projection_matrix(K, rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    t = np.asarray(tvec, dtype=float).reshape(3, 1)
    return K @ np.hstack((R, t))
