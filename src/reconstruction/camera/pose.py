from pathlib import Path
import json
import cv2
import numpy as np

def save_camera(path, K, dist, rvec, tvec, image_size, rms=None):
    data = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": np.asarray(K).tolist(),
        "distortion": np.asarray(dist).reshape(-1).tolist(),
        "rvec": np.asarray(rvec).reshape(3).tolist(),
        "tvec": np.asarray(tvec).reshape(3).tolist(),
    }
    if rms is not None:
        data["rms_reprojection_error_px"] = float(rms)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_camera(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"image_size": tuple(d["image_size"]),
            "K": np.asarray(d["camera_matrix"], float),
            "dist": np.asarray(d["distortion"], float),
            "rvec": np.asarray(d["rvec"], float),
            "tvec": np.asarray(d["tvec"], float),
            "rms": d.get("rms_reprojection_error_px")}

def solve_pnp(points_3d, points_2d, K, dist=None):
    dist = np.zeros(5) if dist is None else dist
    ok, rvec, tvec = cv2.solvePnP(np.asarray(points_3d, float),
                                  np.asarray(points_2d, float), K, dist)
    if not ok:
        raise RuntimeError("solvePnP failed")
    return rvec, tvec
