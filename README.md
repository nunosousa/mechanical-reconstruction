# Mechanical Reconstruction

Python/OpenCV/SciPy tools for reconstructing mechanical geometry from photographs using measured 3D landmarks and parametric CAD models.

Initial target: reverse-engineering the frame geometry of a Casal K181 moped.

## Workflow

1. Build a measured 3D reference model in FreeCAD.
2. Record reliable 3D landmarks in `data/casal_k181/measurements/landmarks.csv`.
3. Click those landmarks in photographs with `click_landmarks.py`.
4. Estimate or load the camera model with `fit_camera.py` / `calibrate_camera.py`.
5. Validate reprojection with `validate_camera.py`.
6. Click corresponding tube centreline points in two views with `click_tube.py`.
7. Triangulate them with `triangulate_tube.py`.
8. Fit an intermediate 3D B-spline with `fit_curve.py`.
9. Inspect the result in FreeCAD and convert it to engineering geometry where appropriate.

The long-term goal is to parameterize unknown frame geometry and fit a small number of geometric parameters against multiple photographs.

## Structure

- `src/reconstruction/` — reusable algorithms
- `scripts/` — command-line entry points
- `cad/` — CAD models
- `data/` — source measurements and photographs
- `results/` — generated outputs
- `notebooks/` — exploratory work
- `tests/` — automated tests
- `docs/` — workflow documentation

## Coordinate convention

Use one right-handed coordinate system for the mechanical model. OpenCV camera coordinates use +Z forward from the camera.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For large Canon RAW files, Git LFS is preferable to normal Git history.

A spline is an intermediate reconstruction, not automatically manufacturing geometry. Validate recovered dimensions against the physical frame.
