"""Persistence for a camera model and pose, plus a small PnP wrapper.

A "camera" on disk bundles everything needed to project a 3D scene point
into pixels:

* ``image_size``  — (width, height) of the photograph in pixels.
* ``K``           — 3x3 intrinsic matrix.
* ``dist``        — OpenCV distortion coefficients (5-element vector).
* ``rvec, tvec``  — Rodrigues rotation vector and translation of the world
                    frame expressed in the camera frame (OpenCV convention).
* ``rms``         — optional reprojection error in pixels, for provenance.

The same JSON layout is written by :func:`save_camera` and read by
:func:`load_camera`, so any script that produces a camera model can feed
downstream scripts like ``triangulate_tube.py``.
"""

from pathlib import Path
import json
import cv2
import numpy as np


def save_camera(path, K, dist, rvec, tvec, image_size, rms=None):
    """Write a camera model to JSON. See module docstring for the schema."""
    data = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": np.asarray(K).tolist(),
        "distortion": np.asarray(dist).reshape(-1).tolist(),
        "rvec": np.asarray(rvec).reshape(3).tolist(),
        "tvec": np.asarray(tvec).reshape(3).tolist(),
    }
    if rms is not None:
        data["rms_reprojection_error_px"] = float(rms)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_camera(path):
    """Read a JSON camera file into a dict of numpy arrays."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "image_size": tuple(d["image_size"]),
        "K": np.asarray(d["camera_matrix"], float),
        "dist": np.asarray(d["distortion"], float),
        "rvec": np.asarray(d["rvec"], float),
        "tvec": np.asarray(d["tvec"], float),
        "rms": d.get("rms_reprojection_error_px"),
    }


def solve_pnp(points_3d, points_2d, K, dist=None):
    """Solve for camera pose given known 3D<->2D correspondences.

    This is the "perspective-n-point" problem: intrinsics are already known
    (from a checkerboard calibration), and we recover only ``rvec, tvec``.
    Use :func:`fit_camera_unknown_focal` instead when the focal length is
    also unknown (e.g. for a historical photograph).
    """
    dist = np.zeros(5) if dist is None else dist
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(points_3d, float),
        np.asarray(points_2d, float),
        K, dist,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    return rvec, tvec
