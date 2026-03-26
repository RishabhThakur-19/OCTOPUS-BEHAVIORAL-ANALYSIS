import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    return cap

def get_video_metadata(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, n, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), n / fps

def extract_features(cap, fps, resize_shape=(224, 224)):
    motion, contours, times = [], [], []
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Cannot read video")
    prev_gray = cv2.cvtColor(cv2.resize(prev, resize_shape), cv2.COLOR_BGR2GRAY)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, resize_shape), cv2.COLOR_BGR2GRAY)
        motion.append(np.sum(cv2.absdiff(prev_gray, gray)))
        prev_gray = gray
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(max((cv2.contourArea(c) for c in cnts), default=0))
        times.append(idx / fps)
        idx += 1
    return np.array(motion), np.array(contours), np.array(times)

def normalize(arr):
    rng = arr.max() - arr.min()
    return np.zeros_like(arr, dtype=float) if rng == 0 else (arr - arr.min()) / rng


# helpers

def _frame(fill=None, h=64, w=64):
    return np.full((h, w, 3), fill, dtype=np.uint8) if fill is not None \
        else np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

def _cap(frames, fps=30.0):
    cap = MagicMock()
    cap.get.side_effect = lambda p: {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: float(len(frames)),
        cv2.CAP_PROP_FRAME_WIDTH: 64.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 64.0,
    }.get(p, 0.0)
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    return cap


# tests

def test_open_video_bad_path():
    with patch("cv2.VideoCapture") as mvc:
        mvc.return_value.isOpened.return_value = False
        with pytest.raises(RuntimeError):
            open_video("bad.mp4")

def test_metadata_duration():
    cap = _cap([_frame()] * 300, fps=30.0)
    fps, n, *_, dur = get_video_metadata(cap)
    assert fps == 30.0 and n == 300
    assert pytest.approx(dur) == 10.0

def test_extract_bad_video():
    cap = MagicMock()
    cap.read.return_value = (False, None)
    with pytest.raises(RuntimeError):
        extract_features(cap, fps=30.0)

def test_extract_shapes():
    m, c, t = extract_features(_cap([_frame()] * 6), fps=30.0)
    assert m.shape == c.shape == t.shape == (5,)

def test_extract_single_frame():
    m, c, t = extract_features(_cap([_frame()]), fps=30.0)
    assert len(m) == 0

def test_no_motion_same_frame():
    f = _frame()
    m, _, _ = extract_features(_cap([f.copy(), f.copy()]), fps=30.0)
    assert m[0] == 0

def test_motion_bigger_change():
    base = _frame(fill=0)
    small = base.copy(); small[0, 0] = [10, 10, 10]
    large = base.copy(); large[:] = [100, 100, 100]
    m_s, _, _ = extract_features(_cap([base.copy(), small]), fps=30.0, resize_shape=(64, 64))
    m_l, _, _ = extract_features(_cap([base.copy(), large]), fps=30.0, resize_shape=(64, 64))
    assert m_l[0] > m_s[0]

def test_no_contour_white_frame():
    w = _frame(fill=255)
    _, c, _ = extract_features(_cap([w.copy(), w.copy()]), fps=30.0, resize_shape=(64, 64))
    assert c[0] == 0

def test_normalize():
    n = normalize(np.array([1.0, 5.0, 10.0]))
    assert n.min() == pytest.approx(0.0) and n.max() == pytest.approx(1.0)

def test_normalize_flat():
    assert np.all(normalize(np.array([5.0, 5.0, 5.0])) == 0.0)