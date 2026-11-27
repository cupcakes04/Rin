"""curate.difference: the frames of one frame set that none of several others holds."""

from __future__ import annotations

import pyarrow as pa

from rin.master import Master
from rin.run import Run


def difference(m: Master, frames: Run, minus: list[Run]) -> Run:
    """A frame set holding the frames of `frames` that no frame set in `minus` holds, unranked.

    E.g. the candidates a teacher hasn't labelled yet, when a rebuilt pool keeps most of the
    old candidates (D-37): label only the new ones, then pass both label runs to `m.dataset`.
    """
    for run in [frames, *minus] if isinstance(minus, list) else [frames]:
        if not isinstance(run, Run) or "frame_sets" not in run.tables:
            raise TypeError(f"difference takes frame sets; got {run!r}")
    if not isinstance(minus, list) or not minus:
        raise ValueError(f"minus must be a non-empty list of frame sets, got {minus!r}")
    ids = [run.id for run in minus]
    if frames.id in ids or len(set(ids)) != len(ids):
        raise ValueError(f"A frame set is listed twice: {frames.id} minus {ids}")

    with m.writer(step="curate.difference", config={}, inputs=[frames.id, *ids]) as w:
        rows = m.sql(
            """SELECT DISTINCT media_id, frame_idx FROM frame_sets WHERE run_id = ?
               EXCEPT SELECT media_id, frame_idx FROM frame_sets WHERE list_contains(?, run_id)
               ORDER BY ALL""",
            [frames.id, ids],
        ).fetchall()
        w.write("frame_sets", pa.table({
            "media_id": [media_id for media_id, _ in rows],
            "frame_idx": pa.array([idx for _, idx in rows], pa.int64()),
            "rank": pa.nulls(len(rows), pa.int64()),
        }))
    return m.run(w.run_id)
