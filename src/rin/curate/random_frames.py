"""curate.random: n frames of a frame set, picked at random, in a ranked order."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from rin.master import Master
from rin.run import Run


def random(m: Master, frames: Run, *, n: int, seed: int) -> Run:
    """A frame set of `n` frames drawn at random from `frames`, without repeats.

    `rank` is the draw order, so the first k frames are themselves a random sample of k (D-28).
    The draw runs over the frames sorted by (media_id, frame_idx); media ids come from file
    content, so the same seed picks the same frames in any store holding the same files.
    """
    for name, value in (("n", n), ("seed", seed)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"random needs a frame set; {frames.id} wrote {frames.tables}")
    total = len(frames)
    if not 0 < n <= total:
        raise ValueError(f"n must be between 1 and the {total} frames of {frames.id}, got {n}")

    with m.writer(step="curate.random", config={"n": n, "seed": seed}, inputs=[frames.id]) as w:
        rows = m.sql(
            "SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ? ORDER BY media_id, frame_idx",
            [frames.id],
        ).fetchall()
        drawn = np.random.default_rng(seed).permutation(len(rows))[:n]
        w.write("frame_sets", pa.table({
            "media_id": [rows[i][0] for i in drawn],
            "frame_idx": pa.array([rows[i][1] for i in drawn], pa.int64()),
            "rank": pa.array(range(n), pa.int64()),
        }))
    return m.run(w.run_id)
