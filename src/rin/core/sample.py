"""core.sample: an ingest run's media -> the frames on screen every 1/fps seconds."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from rin.master import Master
from rin.media import sample_frames
from rin.run import Run


def sample(m: Master, source: Run, *, fps: float) -> Run:
    """A frame set: for each media of `source` (an ingest run), the frame on screen at
    0, 1/fps, 2/fps, ... seconds, using each frame's real timestamp. An image gives its one frame.
    """
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not fps > 0:
        raise ValueError(f"fps must be a positive number, got {fps!r}")
    if "media" not in source.tables:
        raise TypeError(f"sample needs an ingest run; {source.id} wrote {source.tables}")

    with m.writer(step="core.sample", config={"fps": fps}, inputs=[source.id]) as w:
        rows = m.sql(
            """SELECT f.media_id, list(f.pts ORDER BY f.frame_idx) AS pts
               FROM frames f JOIN media d USING (media_id)
               WHERE d.run_id = ? GROUP BY f.media_id ORDER BY f.media_id""",
            [source.id],
        ).fetchall()

        media_ids: list[str] = []
        frame_idx: list[np.ndarray] = []
        for media_id, pts in rows:
            picked = sample_frames(np.asarray(pts), fps)
            media_ids.extend([media_id] * len(picked))
            frame_idx.append(picked)

        w.write("frame_sets", pa.table({
            "media_id": media_ids,
            "frame_idx": pa.array(np.concatenate(frame_idx), pa.int64()),
            "rank": pa.nulls(len(media_ids), pa.int64()),
        }))
    return m.run(w.run_id)
