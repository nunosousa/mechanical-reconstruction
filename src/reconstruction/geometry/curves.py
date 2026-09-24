import numpy as np
from scipy.interpolate import splprep, splev

def fit_bspline(points, smoothing=0.0, samples=200):
    p = np.asarray(points, float)
    if len(p) < 4:
        raise ValueError("At least four points are required.")
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.r_[0.0, np.cumsum(d)]
    if u[-1] == 0:
        raise ValueError("All points are identical.")
    u /= u[-1]
    tck, _ = splprep(p.T, u=u, s=float(smoothing), k=min(3, len(p)-1))
    uu = np.linspace(0, 1, samples)
    return np.asarray(splev(uu, tck)).T, tck
