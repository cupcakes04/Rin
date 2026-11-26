"""curate.union: the frames in any of several frame sets, each once."""

from __future__ import annotations

import pyarrow as pa

from rin.master import Master
from rin.run import Run


def union(m: Master, frame_sets: list[Run]) -> Run:
    """A frame set holding every frame of the given frame sets once, unranked.

    E.g. the test footage sampled from two ingest runs, or the frames several selections picked
    so SAM3 labels them once (D-28).
    """
    if not isinstance(frame_sets, list) or len(frame_sets) < 2:
        raise ValueError(f"union needs a list of at least two frame sets, got {frame_sets!r}")
    for run in frame_sets:
        if not isinstance(run, Run) or "frame_sets" not in run.tables:
            raise TypeError(f"union takes frame sets; got {run!r}")
    ids = [run.id for run in frame_sets]
    if len(set(ids)) != len(ids):
        raise ValueError(f"The same frame set is listed twice: {ids}")

    with m.writer(step="curate.union", config={}, inputs=ids) as w:
        rows = m.sql(
            "SELECT DISTINCT media_id, frame_idx FROM frame_sets WHERE list_contains(?, run_id) ORDER BY ALL",
            [ids],
        ).fetchall()
        w.write("frame_sets", pa.table({
            "media_id": [media_id for media_id, _ in rows],
            "frame_idx": pa.array([idx for _, idx in rows], pa.int64()),
            "rank": pa.nulls(len(rows), pa.int64()),
        }))
    return m.run(w.run_id)
