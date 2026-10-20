import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent

from reconstruction.image import clicking
from reconstruction.image.clicking import fit_ellipse_near_hint


def test_fit_ellipse_near_rough_circle_hint():
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    cv2.circle(image, (220, 170), 42, (255, 255, 255), 2, cv2.LINE_AA)

    center, axes, _ = fit_ellipse_near_hint(image, (224, 167), 39)

    assert np.linalg.norm(center - (220, 170)) < 2
    assert np.allclose(axes, (84, 84), atol=5)


def test_click_points_manually_recovers_failed_circle_fit(tmp_path, monkeypatch):
    image_path = tmp_path / "photo.png"
    assert cv2.imwrite(str(image_path), np.zeros((40, 40, 3), dtype=np.uint8))

    clicks = iter([
        [(10, 10), (15, 10)],
        [(12, 13)],
        [(25, 25)],
    ])
    monkeypatch.setattr(
        clicking, "_collect_clicks",
        lambda _fig, _ax, count, _title, **_kwargs: (
            next(clicks) if count else []))
    monkeypatch.setattr("builtins.input", lambda _prompt: "m")
    monkeypatch.setattr(
        clicking, "fit_ellipse_near_hint",
        lambda *args: (_ for _ in ()).throw(ValueError("no edge found")))

    points, methods = clicking.click_points(
        image_path, ["bearing", "frame joint"], circle_names={"bearing"})

    assert points == [(12, 13), (25, 25)]
    assert methods == ["manual_center", "manual"]


def test_collect_clicks_zoom_drag_is_not_recorded_as_a_point(monkeypatch):
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)))
    fig.canvas.draw()
    sent_events = False

    def send_events(_interval):
        nonlocal sent_events
        if sent_events:
            return
        sent_events = True
        fig.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", fig.canvas, key="z"))

        def mouse_event(name, point):
            pixel = ax.transData.transform(point)
            return MouseEvent(name, fig.canvas, *pixel, button=1)

        fig.canvas.callbacks.process(
            "button_press_event", mouse_event("button_press_event", (1, 1)))
        fig.canvas.callbacks.process(
            "button_release_event", mouse_event("button_release_event", (8, 8)))
        fig.canvas.callbacks.process(
            "button_press_event", mouse_event("button_press_event", (5, 5)))

    monkeypatch.setattr(plt, "pause", send_events)
    try:
        points = clicking._collect_clicks(fig, ax, 1, "Pick point")
    finally:
        plt.close(fig)

    assert np.allclose(points, [(5, 5)], atol=0.05)
    assert np.allclose(ax.get_xlim(), (1, 8), atol=0.05)
    assert np.allclose(ax.get_ylim(), (8, 1), atol=0.05)


def test_circle_fit_selection_method_is_reported(tmp_path, monkeypatch):
    image_path = tmp_path / "photo.png"
    assert cv2.imwrite(str(image_path), np.zeros((40, 40, 3), dtype=np.uint8))
    clicks = iter([[(10, 10), (15, 10)]])
    monkeypatch.setattr(
        clicking, "_collect_clicks",
        lambda _fig, _ax, count, _title, **_kwargs: (
            next(clicks) if count else []))
    monkeypatch.setattr(
        clicking, "fit_ellipse_near_hint",
        lambda *_args: (np.array([11, 12]), np.array([8, 8]), 0))

    points, methods = clicking.click_points(
        image_path, ["bearing"], circle_names={"bearing"})

    assert np.allclose(points, [(11, 12)])
    assert methods == ["circle_fit"]