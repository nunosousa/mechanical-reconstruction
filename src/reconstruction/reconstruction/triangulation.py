import cv2
import numpy as np

def triangulate(points1, points2, P1, P2):
    x1 = np.asarray(points1, float).T
    x2 = np.asarray(points2, float).T
    X = cv2.triangulatePoints(P1, P2, x1, x2)
    X /= X[3:4, :]
    return X[:3, :].T

def reprojection_error(points_3d, points_2d, P):
    X = np.hstack([np.asarray(points_3d), np.ones((len(points_3d), 1))])
    x = (P @ X.T).T
    x = x[:, :2] / x[:, 2:3]
    return np.linalg.norm(x - np.asarray(points_2d), axis=1)
