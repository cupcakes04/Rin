"""Time -> frame: which frame is on screen at a given time.

Pure numpy; knows nothing about files or the store.
"""

from __future__ import annotations

import numpy as np

TICK_TOLERANCE = 5e-4
"""Seconds. Containers store times in ticks, so a frame meant to start at 0.1 s can read back
as 0.1001 s. Half a millisecond absorbs that and is far below any real gap between frames."""


def frame_on_screen(pts: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the frame on screen at each time: the latest frame that started at or before it.

    `pts` are the frames' start times in seconds since frame 0, strictly increasing.
    A time after the last frame's start maps to the last frame (it stays on screen).
    A negative time is an error: nothing is on screen before frame 0.
    """
    pts = np.asarray(pts, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if pts.ndim != 1 or len(pts) == 0:
        raise ValueError(f"pts must be a non-empty 1-D array, got shape {pts.shape}")
    if len(pts) > 1 and not np.all(np.diff(pts) > 0):
        raise ValueError("pts must be strictly increasing")
    if np.any(times < -TICK_TOLERANCE):
        raise ValueError(f"times before frame 0 have no frame on screen: {times[times < 0][:5].tolist()}")
    return np.searchsorted(pts, times + TICK_TOLERANCE, side="right") - 1


def sample_frames(pts: np.ndarray, fps: float) -> np.ndarray:
    """Frames on screen at 0, 1/fps, 2/fps, ... up to the last frame's start; each frame once.

    If `fps` is above the video's own rate, frames repeat in time and are kept once,
    so the result is never longer than the video.
    """
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}")
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("pts is empty")
    times = np.arange(0.0, pts[-1] + TICK_TOLERANCE, 1.0 / fps)
    return np.unique(frame_on_screen(pts, times))
