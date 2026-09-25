"""Two-view triangulation of corresponding image points into 3D.

Step 7 of the workflow: given the same physical point clicked in two
images, and the calibrated pose of each camera, recover the 3D point.
"""

import cv2
import numpy as np


def triangulate(points1, points2, P1, P2):
    """Linear triangulation of matched image points via OpenCV.

    ``P1`` and ``P2`` are 3x4 projection matrices for the two views (see
    ``camera.projection.projection_matrix``). ``points1[i]`` and
    ``points2[i]`` must refer to the same physical point.
    """
    # cv2.triangulatePoints wants each view's points as 2xN.
    x1 = np.asarray(points1, float).T
    x2 = np.asarray(points2, float).T
    X = cv2.triangulatePoints(P1, P2, x1, x2)
    # Convert from homogeneous 4-vectors back to Euclidean 3D.
    X /= X[3:4, :]
    return X[:3, :].T


def reprojection_error(points_3d, points_2d, P):
    """Per-point pixel error after projecting 3D points through P.

    Handy for validating a triangulated point cloud against its original
    2D clicks — a well-triangulated point should re-project close to
    where it was clicked.
    """
    # Append a column of ones to make homogeneous 4-vectors.
    X = np.hstack([np.asarray(points_3d), np.ones((len(points_3d), 1))])
    x = (P @ X.T).T
    x = x[:, :2] / x[:, 2:3]
    return np.linalg.norm(x - np.asarray(points_2d), axis=1)
