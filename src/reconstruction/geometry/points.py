import numpy as np

def pairwise_distances(points):
    p = np.asarray(points, float)
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
