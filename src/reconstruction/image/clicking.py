import cv2
import matplotlib.pyplot as plt

def click_points(image_path, names):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    points = []
    for name in names:
        ax.set_title(f"Click: {name}")
        result = plt.ginput(1, timeout=-1)
        if not result:
            plt.close(fig)
            raise RuntimeError("Point selection cancelled.")
        u, v = result[0]
        points.append((u, v))
        ax.plot(u, v, "r+", markersize=10)
        fig.canvas.draw_idle()
    plt.close(fig)
    return points

def click_polyline(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Click points in order; press Enter when done")
    points = plt.ginput(-1, timeout=-1)
    plt.close(fig)
    return points
