# Camera calibration

Use a checkerboard at multiple orientations and positions, filling a useful portion of the image.

Keep fixed:
- focal length
- focus
- resolution
- cropping
- image processing

If using a tripod, disable image stabilisation if practical.

The calibration produces the intrinsic matrix and lens distortion coefficients. Use the same optical configuration for reconstruction photographs.

Calibration images are loaded in the sensor frame (EXIF orientation ignored), so
landscape and portrait shots of the board can be mixed; all images must still
share one pixel size. Calibrations made before this was enforced may have used
EXIF-rotated portrait images and should be redone (a principal point far from
the image centre or `fx` noticeably different from `fy` is a warning sign).
