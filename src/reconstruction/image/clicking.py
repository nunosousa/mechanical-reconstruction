"""Interactive point picking on top of matplotlib.

Both helpers open the requested image in a matplotlib window and wait for
the user to click. They block until the user is done — timeouts are
disabled on purpose so you can zoom / pan before placing a point.
"""

import cv2
import matplotlib.pyplot as plt


def click_points(image_path, names):
    """Ask the user to click one point per name, in order.

    A crosshair is drawn at each accepted click for visual feedback. The
    returned list of (u, v) pixel coordinates is aligned 1:1 with ``names``.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    # matplotlib expects RGB; OpenCV loaded BGR.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    points = []
    for name in names:
        ax.set_title(f"Click: {name}")
        # timeout=-1 means "wait forever" — the user may take time to zoom.
        result = plt.ginput(1, timeout=-1)
        if not result:
            # Closing the window returns an empty list; treat as cancel.
            plt.close(fig)
            raise RuntimeError("Point selection cancelled.")
        u, v = result[0]
        points.append((u, v))
        ax.plot(u, v, "r+", markersize=10)
        fig.canvas.draw_idle()
    plt.close(fig)
    return points


def click_polyline(image_path):
    """Collect an unordered-length sequence of points, ending on Enter.

    Used for tube centrelines where the number of samples is up to the
    user and depends on how curved the tube looks in that view.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Click points in order; press Enter when done")
    # n=-1 keeps collecting clicks until the user presses Enter.
    points = plt.ginput(-1, timeout=-1)
    plt.close(fig)
    return points
