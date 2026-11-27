"""curate.center: subtract each stream's mean vector (one camera, day and split), so what all
its frames share (the fixed background) drops out and what differs (the people) is left.

On MEVA, two random frames of one camera have a median whole-frame cosine of 0.95; with the
camera's mean removed, about 0.0 (holy grail §4). The result is an ordinary embedding run, so
dedupe and selection take it like any other.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from rin.curate.streams import stream_sql
from rin.master import Master
from rin.run import Run


def center(m: Master, emb: Run) -> Run:
    """A new embedding run: every vector minus the mean of its stream's vectors of the same kind.

    A stream is one camera on one day in one split (`rin.curate.streams`); a media without a
    camera is its own stream. The mean is over the frames in `emb`, so it describes those
    frames' shared background, not the whole recording.
    """
    if "embeddings" not in emb.tables:
        raise TypeError(f"center needs an embedding run; {emb.id} wrote {emb.tables}")
    # model stays null: centring is a rule; the model is on the input run (`inputs`).
    with m.writer(step="curate.center", config={"by": "stream"}, inputs=[emb.id]) as w:
        index = m.sql(
            f"""SELECT e.row, e.media_id, e.frame_idx, e.instance_idx, e.kind, {stream_sql()} AS grp
               FROM embeddings e JOIN media d USING (media_id)
               WHERE e.run_id = ? ORDER BY e.row""",
            [emb.id],
        ).df()
        source = emb.vectors()
        if len(index) != len(source) or not (index["row"].to_numpy() == np.arange(len(source))).all():
            raise ValueError(f"{emb.id}: the embeddings table doesn't index every row of embeddings.npy once")

        centred = np.empty(source.shape, np.float32)
        for (grp, kind), rows in index.groupby(["grp", "kind"])["row"]:
            if len(rows) < 2:
                # One frame minus its own mean is all zeros: nothing is left to compare.
                raise ValueError(f"{emb.id}: {grp} has a single {kind!r} vector; centring needs at least 2")
            block = source[rows.to_numpy()].astype(np.float32)
            centred[rows.to_numpy()] = block - block.mean(axis=0)
        stored = centred.astype(np.float16)
        if not np.isfinite(stored).all():
            raise ValueError(f"{emb.id}: centred vectors overflow float16")

        table = pa.table({
            "row": pa.array(index["row"], pa.int64()),
            "media_id": index["media_id"].tolist(),
            "frame_idx": pa.array(index["frame_idx"], pa.int64()),
            "instance_idx": pa.array(index["instance_idx"].astype("Int64"), pa.int64()),
            "kind": index["kind"].tolist(),
        })
        np.save(w.tmp_dir / "embeddings.npy", stored)
        w.write("embeddings", table)
    return m.run(w.run_id)
