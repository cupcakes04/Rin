"""curate.select: a ranked selection of frames for labelling (D-28, D-38).

Every method writes a frame set whose `rank` orders the picks, so budget N is the first N.
`kcenter` and `kmeans_stratified` work on each stream (one camera, day and split) alone, by
Euclidean distance between raw vectors, and the streams take turns in the merged ranking; so
every budget is spread evenly over the cameras (D-38). `random` draws over all frames at once.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyarrow as pa

from rin.curate.streams import stream_sql
from rin.master import Master
from rin.run import Run

METHODS = ("random", "kcenter", "kmeans_stratified")
KMEANS_ITERATIONS = 100
"""At most this many k-means updates; it stops earlier once no frame changes cluster."""


def select(
    m: Master,
    frames: Run,
    emb: Run,
    *,
    method: str,
    n_max: int,
    seed: int,
    clusters: int | None = None,
    kind: str = "cls",
) -> Run:
    """Rank up to `n_max` frames of frame set `frames` with `method`, using `emb`'s `kind` vectors.

    random:            a random order over all frames (the same draw as `curate.random`).
    kcenter:           per stream, farthest-point: a first frame drawn by `seed`, then again and
                       again the frame farthest from every frame picked so far.
    kmeans_stratified: per stream, k-means into `clusters` groups; then rounds of one frame per
                       cluster, clusters in a random order, frames drawn at random inside a
                       cluster. `clusters` is required for it, and refused for the others.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    for name, value in (("n_max", n_max), ("seed", seed)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if n_max == 0:
        raise ValueError("n_max must be at least 1")
    if method == "kmeans_stratified":
        if not isinstance(clusters, int) or isinstance(clusters, bool) or clusters < 1:
            raise ValueError(f"kmeans_stratified needs clusters, a positive integer, got {clusters!r}")
    elif clusters is not None:
        raise ValueError(f"clusters is only for kmeans_stratified, not {method}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"select needs a frame set; {frames.id} wrote {frames.tables}")
    if "embeddings" not in emb.tables:
        raise TypeError(f"select needs an embedding run; {emb.id} wrote {emb.tables}")
    if not 0 < n_max <= len(frames):
        raise ValueError(f"n_max must be between 1 and the {len(frames)} frames of {frames.id}, got {n_max}")

    config = {"method": method, "n_max": n_max, "seed": seed, "kind": kind}
    if clusters is not None:
        config["clusters"] = clusters
    with m.writer(step="curate.select", config=config, inputs=[frames.id, emb.id]) as w:
        rows = m.sql(
            f"""SELECT s.media_id, s.frame_idx, e.row, {stream_sql()} AS stream
                FROM frame_sets s
                JOIN media d USING (media_id)
                LEFT JOIN (SELECT * FROM embeddings WHERE run_id = ? AND kind = ? AND instance_idx IS NULL) e
                     USING (media_id, frame_idx)
                WHERE s.run_id = ?
                ORDER BY s.media_id, s.frame_idx""",
            [emb.id, kind, frames.id],
        ).fetchall()
        missing = [(media_id, idx) for media_id, idx, row, _ in rows if row is None]
        if missing:
            raise ValueError(f"{emb.id} has no {kind!r} vector for {len(missing)} frames, e.g. {missing[:3]}")

        rng = np.random.default_rng(seed)
        if method == "random":
            # Over (media_id, frame_idx) order, like curate.random: same seed, same frames.
            order = list(rng.permutation(len(rows))[:n_max])
        else:
            vectors = emb.vectors()
            by_stream: dict[str, list[int]] = defaultdict(list)
            for i, (_, _, _, stream) in enumerate(rows):
                by_stream[stream].append(i)
            ranked = []
            for stream in sorted(by_stream):
                members = by_stream[stream]
                x = vectors[[rows[i][2] for i in members]].astype(np.float64)
                n = min(n_max, len(members))
                if method == "kcenter":
                    picks = _kcenter(x, n, rng)
                else:
                    if clusters > len(members):
                        raise ValueError(f"{stream} has {len(members)} frames, fewer than clusters={clusters}")
                    picks = _stratified(_kmeans(x, clusters, rng), n, rng)
                ranked.append([members[p] for p in picks])
            order = _in_turns(ranked)[:n_max]

        w.write("frame_sets", pa.table({
            "media_id": [rows[i][0] for i in order],
            "frame_idx": pa.array([rows[i][1] for i in order], pa.int64()),
            "rank": pa.array(range(len(order)), pa.int64()),
        }))
    return m.run(w.run_id)


def _kcenter(x: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """Farthest-point order of the rows of x: a random first row, then each time the row
    farthest (Euclidean) from all rows picked so far. Ties go to the lowest index."""
    first = int(rng.integers(len(x)))
    picks = [first]
    nearest = np.linalg.norm(x - x[first], axis=1)  # distance to the closest pick so far
    nearest[first] = -1  # never picked twice, even among identical vectors (all distances 0)
    while len(picks) < n:
        nxt = int(nearest.argmax())
        picks.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(x - x[nxt], axis=1))
        nearest[picks] = -1
    return picks


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster of each row of x: k-means on Euclidean distance, started by k-means++ (each new
    centre drawn with probability growing with its squared distance to the nearest centre)."""
    centres = [x[int(rng.integers(len(x)))]]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(x[int(rng.choice(len(x), p=d2 / d2.sum()))] if d2.sum() > 0 else x[int(rng.integers(len(x)))])
    centres = np.array(centres)
    labels = np.full(len(x), -1)
    for _ in range(KMEANS_ITERATIONS):
        d2 = (x**2).sum(axis=1)[:, None] - 2 * x @ centres.T + (centres**2).sum(axis=1)[None, :]
        new = d2.argmin(axis=1)
        if (new == labels).all():
            break
        labels = new
        for c in range(k):
            if (labels == c).any():  # an emptied cluster keeps its old centre
                centres[c] = x[labels == c].mean(axis=0)
    return labels


def _stratified(labels: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """Rounds of one row per cluster, clusters in a random order, rows drawn at random inside
    each cluster; a cluster that runs out is skipped."""
    groups = [list(rng.permutation(np.flatnonzero(labels == c))) for c in rng.permutation(labels.max() + 1)]
    return [int(i) for i in _in_turns(groups)[:n]]


def _in_turns(lists: list[list[int]]) -> list[int]:
    """First items of every list, then second items, …; lists that run out are skipped."""
    out = []
    for r in range(max((len(x) for x in lists), default=0)):
        out.extend(x[r] for x in lists if r < len(x))
    return out
