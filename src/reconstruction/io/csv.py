from pathlib import Path
import csv
import numpy as np

def read_landmarks_3d(path):
    names, points = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            points.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return names, np.asarray(points, dtype=float)

def read_points_2d(path):
    names, points = [], []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            points.append([float(row["u"]), float(row["v"])])
    return names, np.asarray(points, dtype=float)

def write_points_2d(path, names, points):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "u", "v"])
        w.writerows([n, float(p[0]), float(p[1])] for n, p in zip(names, points))

def write_points_3d(path, points, names=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = names or [f"P{i:03d}" for i in range(len(points))]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "x", "y", "z"])
        w.writerows([n, float(p[0]), float(p[1]), float(p[2])]
                    for n, p in zip(names, points))
