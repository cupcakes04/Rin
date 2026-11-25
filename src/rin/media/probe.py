"""Read a media file's size and every frame's timestamp.

The only place in Rin that decides what a frame index means: frame i is the i-th frame the
decoder returns. Knows nothing about the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np

VIDEO_SUFFIXES = frozenset({".avi", ".mp4", ".mkv", ".mov", ".webm", ".m4v"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True)
class Probe:
    kind: Literal["video", "image"]
    width: int
    height: int
    pts: np.ndarray
    """Start time of every frame in seconds since frame 0. An image has one frame at 0.0."""


def media_kind(path: Path) -> Literal["video", "image"]:
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    raise ValueError(f"Not a video or image Rin reads: {path} (suffix {suffix!r})")


def probe(path: Path) -> Probe:
    """Decode the file once and return its size and per-frame timestamps.

    Every frame is decoded, not just counted from the container's index, because the
    index can disagree with what a decoder returns, and frame_idx must mean the latter.

    When the header states a frame count (AVI, MP4), the decoded count must equal it: a file
    cut short (an interrupted copy) still opens and decodes, just with fewer frames, and would
    otherwise go into the store as a shorter video. MKV and images state no count, so for them
    this check can't run.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    kind = media_kind(path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        declared = stream.frames  # from the header; 0 when the container doesn't say
        times: list[float] = []
        width = height = 0
        for i, frame in enumerate(container.decode(stream)):
            if i == 0:
                width, height = frame.width, frame.height
            if kind == "image":
                times.append(0.0)
                break
            if frame.time is None:
                raise ValueError(f"{path}: frame {i} has no timestamp")
            times.append(frame.time)
    if not times:
        raise ValueError(f"{path}: the decoder returned no frames")
    if kind == "video" and declared and declared != len(times):
        raise ValueError(
            f"{path}: the header says {declared} frames but the decoder returned {len(times)}; "
            "the file is probably truncated or corrupt"
        )
    # The decoder returns frames in display order, but some containers attach timestamps in
    # storage order: AVI with B-frames gives 1, 4, 3, 5, 2, ... (MEVA does this). The set of
    # timestamps is still right, so the k-th frame shown gets the k-th smallest. For MP4/MKV,
    # whose timestamps already increase, sorting changes nothing.
    pts = np.sort(np.asarray(times, dtype=np.float64))
    pts -= pts[0]
    if len(pts) > 1 and not np.all(np.diff(pts) > 0):
        bad = int(np.argmax(np.diff(pts) <= 0))
        raise ValueError(f"{path}: two frames share a timestamp (frame {bad + 1}); can't tell when each is shown")
    return Probe(kind=kind, width=width, height=height, pts=pts)
