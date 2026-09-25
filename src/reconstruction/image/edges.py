"""Edge detection helpers.

These are stepping stones towards automated tube-centreline extraction:
a distance transform of the detected edges gives, at every pixel, the
distance to the nearest silhouette edge, which can be used as a cost
function when snapping a hand-clicked polyline onto real tube edges.
"""

import cv2


def canny_edges(image, low=50, high=150):
    """Grayscale + Canny edge detection with sensible defaults."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)


def distance_transform_from_edges(edges):
    """Signed distance from every pixel to the nearest edge pixel.

    ``cv2.distanceTransform`` measures distance to the nearest *zero*
    pixel; we invert the Canny output so that edges are the zeros. The
    result is a floating-point image where value ``d`` at pixel ``(y, x)``
    is the Euclidean distance from that pixel to the closest edge.
    """
    mask = (edges == 0).astype("uint8")
    return cv2.distanceTransform(mask, cv2.DIST_L2, 3)
