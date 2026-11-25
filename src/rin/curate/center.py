"""curate.center: subtract each camera's mean vector, so what all its frames share (the fixed
background) drops out and what differs (the people) is left.

On MEVA, two random frames of one camera have a median whole-frame cosine of 0.95; with the
camera's mean removed, about 0.0 (holy grail §4). The result is an ordinary embedding run, so
dedupe and selection take it like any other.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from rin.master import Master
from rin.run import Run


def center(m: Master, emb: Run) -> Run:
    """A new embedding run: every vector minus the mean of its camera's vectors of the same kind.

    A camera is the media's `camera`; a media without one is its own group. The mean is over the
    frames in `emb`, so it describes those frames' shared background, not the whole recording.
    """
    if "embeddings" not in emb.tables:
        raise TypeError(f"center needs an embedding run; {emb.id} wrote {emb.tables}")
    index = m.sql(
        """SELECT e.row, e.media_id, e.frame_idx, e.instance_idx, e.kind,
                  coalesce('camera ' || d.camera, 'media ' || d.media_id) AS grp
           FROM embeddings e JOIN media d USING (media_id)
           WHERE e.run_id = ? ORDER BY e.row""",
        [emb.id],
    ).df()
    source = emb.vectors()
    if len(index) != len(source) or not (index["row"].to_numpy() == np.arange(len(source))).all():
        raise ValueError(f"{emb.id}: the embeddings table doesn't index every row of embeddings.npy once")

    centred = np.empty(source.shape, np.float32)
    for _, rows in index.groupby(["grp", "kind"])["row"]:
        block = source[rows.to_numpy()].astype(np.float32)
        centred[rows.to_numpy()] = block - block.mean(axis=0)

    table = pa.table({
        "row": pa.array(index["row"], pa.int64()),
        "media_id": index["media_id"].tolist(),
        "frame_idx": pa.array(index["frame_idx"], pa.int64()),
        "instance_idx": pa.array(index["instance_idx"].astype("Int64"), pa.int64()),
        "kind": index["kind"].tolist(),
    })
    with m.writer(step="curate.center", config={"by": "camera"}, inputs=[emb.id], model=emb.info["model"]) as w:
        np.save(w.tmp_dir / "embeddings.npy", centred.astype(np.float16))
        w.write("embeddings", table)
    return m.run(w.run_id)
