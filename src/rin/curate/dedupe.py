"""curate.dedupe: drop frames that look like the frame kept just before them."""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from rin.master import Master
from rin.run import Run


def dedupe(m: Master, frames: Run, emb: Run, *, cosine_max: float, kind: str = "cls") -> Run:
    """Walk each camera's frames in time order; keep a frame only if its embedding's cosine
    similarity to the last kept frame is below `cosine_max`.

    A camera's clips are walked as one stream when the media rows know the camera and the
    start time (`describe` at ingest); otherwise each media is walked on its own.
    Comparing with the last kept frame (not every kept frame) removes runs of near-copies, such
    as a person standing still for a minute, and lets a scene that returns later count again:
    spreading the budget over the whole pool is selection's job, not this step's.
    """
    if not isinstance(cosine_max, (int, float)) or isinstance(cosine_max, bool) or not -1 < cosine_max < 1:
        raise ValueError(f"cosine_max must be between -1 and 1, got {cosine_max!r}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"dedupe needs a frame set; {frames.id} wrote {frames.tables}")
    if "embeddings" not in emb.tables:
        raise TypeError(f"dedupe needs an embedding run; {emb.id} wrote {emb.tables}")

    config = {"cosine_max": cosine_max, "kind": kind}
    with m.writer(step="curate.dedupe", config=config, inputs=[frames.id, emb.id]) as w:
        rows = m.sql(
            """SELECT s.media_id, s.frame_idx, e.row,
                      CASE WHEN d.camera IS NOT NULL AND d.started_at IS NOT NULL
                           THEN 'camera ' || d.camera ELSE 'media ' || d.media_id END AS stream
               FROM frame_sets s
               JOIN media d USING (media_id)
               JOIN frames f USING (media_id, frame_idx)
               LEFT JOIN (SELECT * FROM embeddings WHERE run_id = ? AND kind = ? AND instance_idx IS NULL) e
                    USING (media_id, frame_idx)
               WHERE s.run_id = ?
               ORDER BY stream, d.started_at, s.media_id, f.pts""",
            [emb.id, kind, frames.id],
        ).fetchall()
        missing = [(media_id, idx) for media_id, idx, row, _ in rows if row is None]
        if missing:
            raise ValueError(f"{emb.id} has no {kind!r} vector for {len(missing)} frames, e.g. {missing[:3]}")

        vectors = emb.vectors().astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        zero = [(media_id, idx) for media_id, idx, row, _ in rows if norms[row, 0] == 0]
        if zero:
            # A zero vector has no direction, so no cosine; NaN would silently drop the rest of its stream.
            raise ValueError(
                f"{emb.id} has an all-zero {kind!r} vector for {len(zero)} frames, e.g. {zero[:3]}; "
                "cosine similarity is undefined for them"
            )
        vectors /= norms
        kept: list[tuple[str, int]] = []
        last: np.ndarray | None = None
        stream_of_last = None
        for media_id, idx, row, stream in rows:
            v = vectors[row]
            if stream != stream_of_last or last is None or float(v @ last) < cosine_max:
                kept.append((media_id, idx))
                last, stream_of_last = v, stream

        kept.sort()
        table = pa.table({
            "media_id": [media_id for media_id, _ in kept],
            "frame_idx": pa.array([idx for _, idx in kept], pa.int64()),
            "rank": pa.nulls(len(kept), pa.int64()),
        })
        w.write("frame_sets", table)
    return m.run(w.run_id)
