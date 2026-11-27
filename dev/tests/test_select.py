"""curate.select on generated media with hand-made vectors (CPU)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from conftest import CFR_STARTS_MS, make_clip

from rin import Master, Run, core, curate


@pytest.fixture
def frames(m: Master, workspace: Path) -> Run:
    """Camera A: 16 frames; camera B: 12 frames; one day, one split."""
    folder = workspace / "media" / "two_cameras"
    folder.mkdir()
    make_clip(folder / "A.mkv", CFR_STARTS_MS, rate=10)
    make_clip(folder / "B.mkv", CFR_STARTS_MS[:12], rate=10)

    def describe(path: Path) -> dict:
        return {"site": "gym", "camera": path.stem, "started_at": datetime(2018, 3, 7, 10, 0)}

    return core.sample(m, core.ingest(m, folder, split="pool", describe=describe), fps=10)


def _keys(m: Master, frames: Run) -> list[tuple[str, int, str]]:
    """(media_id, frame_idx, camera) of every frame, in (camera, frame_idx) order."""
    return m.sql(
        """SELECT s.media_id, s.frame_idx, d.camera FROM frame_sets s JOIN media d USING (media_id)
           WHERE s.run_id = ? ORDER BY d.camera, s.frame_idx""",
        [frames.id],
    ).fetchall()


def _embedding(m: Master, frames: Run, vectors: dict[tuple[str, int], list[float]]) -> Run:
    """A stand-in for dinov3.embed: the given CLS vector for each (camera, frame_idx)."""
    keys = _keys(m, frames)
    with m.writer(step="dinov3.embed", config={}, inputs=[frames.id]) as w:
        np.save(w.tmp_dir / "embeddings.npy", np.array([vectors[(c, i)] for _, i, c in keys], np.float16))
        w.write("embeddings", pa.table({
            "row": pa.array(range(len(keys)), pa.int64()),
            "media_id": [k[0] for k in keys],
            "frame_idx": pa.array([k[1] for k in keys], pa.int64()),
            "instance_idx": pa.nulls(len(keys), pa.int64()),
            "kind": ["cls"] * len(keys),
        }))
    return m.run(w.run_id)


def _picks(m: Master, sel: Run) -> list[tuple[str, int]]:
    """(camera, frame_idx) of a selection, in rank order."""
    return [tuple(r) for r in m.sql(
        """SELECT d.camera, s.frame_idx FROM frame_sets s JOIN media d USING (media_id)
           WHERE s.run_id = ? ORDER BY s.rank""",
        [sel.id],
    ).fetchall()]


def _on_a_line(m: Master, frames: Run) -> Run:
    """Frame i of each camera sits at distance i along one axis (B shifted far away)."""
    return _embedding(m, frames, {(c, i): [i + (100 if c == "B" else 0), 0] for _, i, c in _keys(m, frames)})


def test_random_is_the_same_draw_as_curate_random(m: Master, frames: Run) -> None:
    emb = _on_a_line(m, frames)
    sel = curate.select(m, frames, emb, method="random", n_max=10, seed=3)
    assert sel.frame_set().df()[["media_id", "frame_idx"]].values.tolist() == (
        curate.random(m, frames, n=10, seed=3).frame_set().df()[["media_id", "frame_idx"]].values.tolist()
    )
    assert sel.config == {"method": "random", "n_max": 10, "seed": 3, "kind": "cls"}


def test_kcenter_picks_the_farthest_frame_each_time_per_camera_in_turns(m: Master, frames: Run) -> None:
    sel = curate.select(m, frames, _on_a_line(m, frames), method="kcenter", n_max=20, seed=0)
    picks = _picks(m, sel)
    assert [c for c, _ in picks[:8]] == ["A", "B"] * 4  # cameras take turns
    for cam, last in (("A", 15), ("B", 11)):
        order = [i for c, i in picks if c == cam]
        for k in range(1, len(order)):  # each pick is a farthest frame from the earlier ones
            dist = {i: min(abs(i - p) for p in order[:k]) for i in range(last + 1) if i not in order[:k]}
            assert dist[order[k]] == max(dist.values())
    assert len(set(picks)) == len(picks) == 20


def test_kcenter_never_repeats_a_frame_among_identical_vectors(m: Master, frames: Run) -> None:
    emb = _embedding(m, frames, {(c, i): [1.0, 1.0] for _, i, c in _keys(m, frames)})
    picks = _picks(m, curate.select(m, frames, emb, method="kcenter", n_max=28, seed=1))
    assert len(set(picks)) == 28


def test_kmeans_stratified_takes_one_frame_per_cluster_first(m: Master, frames: Run) -> None:
    """Four tight blobs per camera: the first round holds one frame of each blob."""
    blob = {i: i // 4 for i in range(16)}
    emb = _embedding(m, frames, {(c, i): [10 * blob[i] + 0.01 * i, 0] for _, i, c in _keys(m, frames)})
    sel = curate.select(m, frames, emb, method="kmeans_stratified", n_max=28, seed=2, clusters=4)
    picks = _picks(m, sel)
    first_round = picks[:8]  # 4 clusters per camera, cameras in turns
    assert sorted(blob[i] for c, i in first_round if c == "A") == [0, 1, 2, 3]
    # B has 12 frames, so only 3 blobs: one of its 4 clusters is half a blob.
    assert {blob[i] for c, i in first_round if c == "B"} == {0, 1, 2}
    assert len(set(picks)) == 28 and sel.config["clusters"] == 4


def test_the_same_seed_picks_the_same_frames(m: Master, frames: Run) -> None:
    emb = _on_a_line(m, frames)
    for method, extra in (("kcenter", {}), ("kmeans_stratified", {"clusters": 3})):
        a = curate.select(m, frames, emb, method=method, n_max=12, seed=5, **extra)
        b = curate.select(m, frames, emb, method=method, n_max=12, seed=5, **extra)
        assert _picks(m, a) == _picks(m, b)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"method": "coreset"}, "method must be"),
        ({"method": "kcenter", "clusters": 4}, "only for kmeans_stratified"),
        ({"method": "kmeans_stratified"}, "needs clusters"),
        ({"method": "random", "n_max": 29}, "n_max must be between"),
        ({"method": "random", "seed": True}, "seed must be"),
    ],
)
def test_select_checks_arguments(m: Master, frames: Run, kwargs: dict, error: str) -> None:
    emb = _on_a_line(m, frames)
    with pytest.raises(ValueError, match=error):
        curate.select(m, frames, emb, **{"n_max": 5, "seed": 0, **kwargs})
