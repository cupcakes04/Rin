"""The frame cache (D-29): the frames a servant needs, decoded once into JPEGs in the store.

Servants read these files and never decode video, so every model sees the same pixels and no
model environment needs a video decoder. Derived data: delete `store/frames/` and it rebuilds.
Each JPEG is pushed to disk before it is renamed into place, so a crash never leaves an empty one.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

from PIL import Image

from rin.master import Master
from rin.media import Rgb, read_frames
from rin.run import Run
from rin.store import write_durably

JPEG_QUALITY = 95
"""A second lossy compression on top of H.264 (D-29's known cost); 95 keeps that loss small."""


def cached_frames(m: Master, frames: Run) -> list[tuple[str, int, Path]]:
    """(media_id, frame_idx, JPEG path) for every frame of a frame set, ordered by media_id and
    frame_idx. Frames not cached yet are decoded first: one pass per media, at full size."""
    if "frame_sets" not in frames.tables:
        raise TypeError(f"Caching frames needs a frame set; {frames.id} wrote {frames.tables}")
    per_media = m.sql(
        """SELECT d.media_id, d.path, list(s.frame_idx ORDER BY s.frame_idx)
           FROM frame_sets s JOIN media d USING (media_id)
           WHERE s.run_id = ? GROUP BY ALL ORDER BY d.media_id""",
        [frames.id],
    ).fetchall()

    out: list[tuple[str, int, Path]] = []
    for media_id, path, idx in per_media:
        files = [m.layout.frame_file(media_id, i) for i in idx]
        # An empty file is a frame a crash cut off before older Rin pushed it to disk: decode it again.
        missing = [i for i, f in zip(idx, files, strict=True) if not f.exists() or f.stat().st_size == 0]
        if missing:
            started = time.perf_counter()
            files[0].parent.mkdir(parents=True, exist_ok=True)
            for i, rgb in read_frames(m.layout.media_file(path), missing):
                _write_jpeg(m.layout.frame_file(media_id, i), rgb)
            print(f"cached {len(missing)} frames of {path} in {time.perf_counter() - started:.1f} s")
        out += [(media_id, i, f) for i, f in zip(idx, files, strict=True)]

    stray = sorted(str(p) for media_id, _, _ in per_media for p in (m.layout.frames / media_id).glob("*.partial"))
    if stray:
        # Left by a process that died mid-write; a cache in use by another process has them too.
        warnings.warn(
            f"{len(stray)} unfinished frame file(s) in the cache, e.g. {stray[:3]}. "
            "If no other process is caching frames, delete them.",
            stacklevel=2,
        )
    return out


def _write_jpeg(target: Path, rgb: Rgb) -> None:
    write_durably(target, lambda partial: Image.fromarray(rgb).save(partial, format="JPEG", quality=JPEG_QUALITY))
