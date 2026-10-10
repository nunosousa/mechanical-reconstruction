"""Interactive point picking on top of matplotlib.

Both helpers open the requested image in a matplotlib window and wait for
the user to click. They block until the user is done — timeouts are
disabled on purpose so you can zoom / pan before placing a point.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def fit_ellipse_near_hint(image, center, radius):
    """Fit the most plausible edge ellipse near a rough center/radius hint."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)

    x, y = center
    extent = int(np.ceil(radius * 1.8))
    left = max(0, int(x) - extent)
    right = min(image.shape[1], int(x) + extent + 1)
    top = max(0, int(y) - extent)
    bottom = min(image.shape[0], int(y) + extent + 1)
    contours, _ = cv2.findContours(
        edges[top:bottom, left:right], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for contour in contours:
        if len(contour) < 12:
            continue
        (local_x, local_y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        axes = np.array([axis_a, axis_b], dtype=float)
        mean_radius = float(axes.mean() / 2)
        ellipse_center = np.array([local_x + left, local_y + top])
        center_error = float(np.linalg.norm(ellipse_center - center))
        if not 0.5 * radius <= mean_radius <= 1.5 * radius:
            continue
        if center_error > 0.75 * radius or min(axes) / max(axes) < 0.35:
            continue

        points = contour[:, 0, :].astype(float)
        points += np.array([left, top])
        radians = np.deg2rad(angle)
        cosine, sine = np.cos(radians), np.sin(radians)
        delta = points - ellipse_center
        along_a = cosine * delta[:, 0] + sine * delta[:, 1]
        along_b = -sine * delta[:, 0] + cosine * delta[:, 1]
        normalized = np.sqrt((2 * along_a / axes[0]) ** 2 +
                             (2 * along_b / axes[1]) ** 2)
        fit_error = float(np.sqrt(np.mean((normalized - 1) ** 2)))
        score = (center_error / radius +
                 abs(np.log(mean_radius / radius)) + 2 * fit_error)
        candidates.append((score, ellipse_center, axes, angle))

    if not candidates:
        raise ValueError("No circular edge found near the supplied hint.")
    _, ellipse_center, axes, angle = min(candidates, key=lambda item: item[0])
    return ellipse_center, axes, angle


def click_points(image_path, names, circle_names=()):
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
    circle_names = set(circle_names)
    for name in names:
        if name in circle_names:
            ax.set_title(f"{name}: click approximate center, then radius")
            result = plt.ginput(2, timeout=-1)
        else:
            ax.set_title(f"Click: {name}")
            # timeout=-1 means "wait forever" — the user may take time to zoom.
            result = plt.ginput(1, timeout=-1)
        if not result:
            # Closing the window returns an empty list; treat as cancel.
            plt.close(fig)
            raise RuntimeError("Point selection cancelled.")
        if name in circle_names:
            if len(result) != 2:
                plt.close(fig)
                raise RuntimeError("Circle selection cancelled.")
            hint_center = np.asarray(result[0], dtype=float)
            hint_radius = float(np.linalg.norm(np.asarray(result[1]) - hint_center))
            if hint_radius < 3:
                plt.close(fig)
                raise ValueError(f"Radius hint for {name} is too small.")
            try:
                fitted_center, axes, angle = fit_ellipse_near_hint(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR), hint_center, hint_radius)
            except ValueError as error:
                plt.close(fig)
                raise ValueError(f"Could not fit {name}: {error}") from error
            u, v = fitted_center
            ax.add_patch(Ellipse((u, v), axes[0], axes[1], angle=angle,
                                 fill=False, edgecolor="lime", linewidth=1.5))
        else:
            u, v = result[0]
        points.append((u, v))
        ax.plot(u, v, "r+", markersize=10)
        if name in circle_names:
            ax.set_title(f"{name}: fitted center shown; click next landmark")
        fig.canvas.draw_idle()
    if circle_names:
        ax.set_title("Review fitted circles; press Enter to finish")
        plt.ginput(0, timeout=-1)
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
