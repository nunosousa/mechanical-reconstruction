# Reconstruction workflow

Direct physical measurements are the geometric reference.

For controlled photographs, calibrate the Canon EOS M50 at the focal length used for reconstruction. Keep focal length, focus, resolution, cropping and image processing fixed.

For historical photographs without calibration, estimate focal length and pose jointly from measured 3D landmarks. Treat this as an estimated camera model and validate reprojection before using it for triangulation.

Trace tube centreline points manually in two sufficiently different views. Do not assume a tube is planar just because it appears planar in one image.

Triangulate corresponding points and compare the recovered geometry against known frame dimensions. Fit a spline only as an intermediate representation.

For final engineering reconstruction, inspect curvature and replace the spline with straight sections, circular arcs and constant-radius bends where justified.

For comparison with the historical target frame, parameterize only the unknown geometry and fit those few parameters against several photographs.
