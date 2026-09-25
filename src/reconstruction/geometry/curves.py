"""Fitting parametric curves through 3D points.

Used at step 8 of the workflow: after triangulating a tube's centreline
we get an ordered but noisy set of 3D points; a smooth B-spline gives a
usable intermediate representation before we replace it with straight
segments and circular arcs in FreeCAD.
"""

import numpy as np
from scipy.interpolate import splprep, splev


def fit_bspline(points, smoothing=0.0, samples=200):
    """Fit a 3D cubic B-spline through an ordered sequence of points.

    Parameters
    ----------
    points : (N, 3) array-like
        Ordered 3D points along the curve.
    smoothing : float
        SciPy ``s`` parameter. ``0`` forces interpolation; larger values
        smooth through noise at the cost of following the points less
        exactly. Increase this when triangulated points are visibly jittery.
    samples : int
        Number of evenly-parameterised samples to return along the curve.

    Returns
    -------
    curve : (samples, 3) ndarray
        Sampled points along the fitted spline.
    tck : tuple
        SciPy's spline representation, in case the caller wants to
        evaluate derivatives or reparameterise.
    """
    p = np.asarray(points, float)
    if len(p) < 4:
        # Need at least four points for a cubic spline.
        raise ValueError("At least four points are required.")

    # Chord-length parameterisation: the spline parameter u advances in
    # proportion to Euclidean distance between successive points, which
    # avoids the over/undershoot you get from uniform parameterisation
    # when point spacing is uneven.
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.r_[0.0, np.cumsum(d)]
    if u[-1] == 0:
        raise ValueError("All points are identical.")
    u /= u[-1]

    tck, _ = splprep(p.T, u=u, s=float(smoothing), k=min(3, len(p) - 1))
    uu = np.linspace(0, 1, samples)
    # splev returns a list of per-axis arrays; stack to (samples, 3).
    return np.asarray(splev(uu, tck)).T, tck
