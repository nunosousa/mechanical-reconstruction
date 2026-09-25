"""Small geometric utilities on point sets."""

import numpy as np


def pairwise_distances(points):
    """Return the symmetric matrix of Euclidean distances between rows.

    Useful for sanity-checking a set of measured landmarks against known
    physical dimensions of the frame.
    """
    p = np.asarray(points, float)
    # Broadcasting trick: p[:, None, :] - p[None, :, :] is (N, N, 3).
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
