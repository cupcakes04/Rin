"""Reading chosen frames out of a media file: the only video decoding steps and servants use.

One pass, front to back, with the same decoding probe() uses to number frames, so the frame
handed over as k is the frame probe numbered k. It never seeks: seeking to a frame index lands
on the nearest keyframe and counts from there, and with B-frames (MEVA's AVI files) that count
can be off. Knows nothing about the store.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
from jaxtyping import Int, UInt8

from rin.media.probe import media_kind

Rgb = UInt8[np.ndarray, "height width 3"]


def read_frames(
    path: Path,
    frame_idx: Int[np.ndarray, " n"] | list[int],
    *,
    size: tuple[int, int] | None = None,
) -> Iterator[tuple[int, Rgb]]:
    """Yield (frame_idx, RGB pixels) for each wanted frame, in increasing frame order.

    frame_idx: the frames wanted, strictly increasing.
    size:      (width, height) to scale to while converting (cheap: done by FFmpeg), or None
               for the file's own size.

    Every frame is decoded, because that is the only way to count them as probe does; only
    the wanted ones are converted to RGB. Raises if the file ends before the last wanted frame.
    """
    path = Path(path)
    wanted = np.asarray(frame_idx, dtype=np.int64)
    if wanted.ndim != 1 or len(wanted) == 0:
        raise ValueError(f"frame_idx must be a non-empty list of frame numbers, got shape {wanted.shape}")
    if wanted[0] < 0 or (len(wanted) > 1 and not np.all(np.diff(wanted) > 0)):
        raise ValueError("frame_idx must be non-negative and strictly increasing")
    if size is not None and (len(size) != 2 or min(size) < 1):
        raise ValueError(f"size must be (width, height) in pixels, got {size!r}")
    scale = {} if size is None else {"width": int(size[0]), "height": int(size[1])}
    kind = media_kind(path)
    if kind == "image" and wanted.tolist() != [0]:
        raise ValueError(f"{path} is an image: its only frame is 0, asked for {wanted.tolist()}")

    targets = iter(wanted.tolist())
    target = next(targets)
    decoded = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # as in probe(): frames still come out in display order
        for i, frame in enumerate(container.decode(stream)):
            decoded = i + 1
            if i == target:
                yield i, frame.to_ndarray(format="rgb24", **scale)
                target = next(targets, None)
                if target is None:
                    return
    raise ValueError(f"{path}: wanted frame {target}, but the decoder returned only {decoded} frames")
