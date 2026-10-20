"""Interactive point picking on top of matplotlib.

The pickers open the requested image in a matplotlib window and wait for the
user to click. Landmark picking supports drag-rectangle zoom before placing
points; timeouts are disabled so you can take your time.
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


def _fit_circle_with_fallback(
        image, fig, ax, name, hint_center, hint_radius, reset_view):
    """Fit a circle hint, offering interactive recovery when fitting fails."""
    while True:
        try:
            if hint_radius < 3:
                raise ValueError(f"Radius hint for {name} is too small.")
            center, axes, angle = fit_ellipse_near_hint(
                image, hint_center, hint_radius)
            return center, axes, angle, "circle_fit"
        except ValueError as error:
            print(f"Could not fit {name}: {error}")
            choice = input(
                "[r]etry with a new hint, [m]anually click center, "
                "[u]se approximate center: ").strip().lower()
            if choice == "m":
                result = _collect_clicks(
                    fig, ax, 1, f"{name}: click the center manually",
                    reset_view=reset_view)
                if not result:
                    raise RuntimeError("Manual circle center selection cancelled.")
                return np.asarray(result[0], dtype=float), None, None, "manual_center"
            if choice == "u":
                return hint_center, None, None, "approximate_center"
            if choice == "r":
                result = _collect_clicks(
                    fig, ax, 2,
                    f"{name}: click a new approximate center, then radius",
                    reset_view=reset_view)
                if not result:
                    raise RuntimeError("Circle selection cancelled.")
                if len(result) != 2:
                    raise RuntimeError("Circle selection cancelled.")
                hint_center = np.asarray(result[0], dtype=float)
                hint_radius = float(np.linalg.norm(
                    np.asarray(result[1]) - hint_center))
                continue
            print("Choose 'r', 'm', or 'u'.")


def _collect_clicks(fig, ax, count, title, reset_view=None):
    """Collect clicks, with a drag-rectangle zoom mode that ignores clicks."""
    points = []
    zooming = False
    zoom_start = None
    finished = False
    if reset_view is None:
        original_xlim, original_ylim = ax.get_xlim(), ax.get_ylim()
    else:
        original_xlim, original_ylim = reset_view
    toolbar = getattr(getattr(fig.canvas, "manager", None), "toolbar", None)

    def set_title():
        controls = (
            "drag rectangle to zoom; r: reset"
            if zooming else "z: zoom; r: reset")
        if count == 0:
            controls += "; Enter: finish"
        ax.set_title(f"{title} ({controls})")

    def toolbar_active():
        mode = getattr(toolbar, "mode", "")
        return bool(getattr(mode, "value", mode))

    def on_press(event):
        nonlocal zoom_start
        if event.inaxes != ax or event.button != 1:
            return
        if toolbar_active():
            return
        if zooming:
            if event.xdata is not None and event.ydata is not None:
                zoom_start = (event.xdata, event.ydata)
        elif count and event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))

    def on_release(event):
        nonlocal zooming, zoom_start
        if not zooming or zoom_start is None:
            return
        start = zoom_start
        zoom_start = None
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        left, right = sorted((start[0], event.xdata))
        bottom, top = sorted((start[1], event.ydata))
        if left == right or bottom == top:
            return
        ax.set_xlim(left, right)
        ax.set_ylim((top, bottom) if ax.yaxis_inverted() else (bottom, top))
        zooming = False
        set_title()
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal zooming, finished, zoom_start
        if event.key == "z":
            zooming = not zooming
            zoom_start = None
            set_title()
        elif event.key == "r":
            zooming = False
            zoom_start = None
            ax.set_xlim(original_xlim)
            ax.set_ylim(original_ylim)
            set_title()
            fig.canvas.draw_idle()
        elif event.key == "escape" or (event.key == "enter" and count == 0):
            finished = True

    set_title()
    connections = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
    ]
    try:
        while not finished and (count == 0 or len(points) < count):
            if not plt.fignum_exists(fig.number):
                break
            plt.pause(0.01)
    finally:
        for connection in connections:
            fig.canvas.mpl_disconnect(connection)
    return points


def click_points(image_path, names, circle_names=()):
    """Ask the user to click one point per name, in order.

    A crosshair is drawn at each accepted click for visual feedback. The
    returned coordinates and selection methods are aligned 1:1 with ``names``.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    # matplotlib expects RGB; OpenCV loaded BGR.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    reset_view = (ax.get_xlim(), ax.get_ylim())
    points = []
    methods = []
    circle_names = set(circle_names)
    try:
        for name in names:
            if name in circle_names:
                result = _collect_clicks(
                    fig, ax, 2, f"{name}: click approximate center, then radius",
                    reset_view=reset_view)
            else:
                result = _collect_clicks(
                    fig, ax, 1, f"Click: {name}", reset_view=reset_view)
            if not result:
                raise RuntimeError("Point selection cancelled.")
            if name in circle_names:
                if len(result) != 2:
                    raise RuntimeError("Circle selection cancelled.")
                hint_center = np.asarray(result[0], dtype=float)
                hint_radius = float(np.linalg.norm(
                    np.asarray(result[1]) - hint_center))
                fitted_center, axes, angle, method = _fit_circle_with_fallback(
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR), fig, ax, name,
                    hint_center, hint_radius, reset_view)
                u, v = fitted_center
                if axes is not None:
                    ax.add_patch(Ellipse((u, v), axes[0], axes[1], angle=angle,
                                         fill=False, edgecolor="lime", linewidth=1.5))
            else:
                u, v = result[0]
                method = "manual"
            points.append((u, v))
            methods.append(method)
            ax.plot(u, v, "r+", markersize=10)
            if name in circle_names:
                ax.set_title(f"{name}: center shown; click next landmark")
            fig.canvas.draw_idle()
        if circle_names:
            _collect_clicks(
                fig, ax, 0, "Review selected centers; press Enter to finish",
                reset_view=reset_view)
        return points, methods
    finally:
        plt.close(fig)


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
