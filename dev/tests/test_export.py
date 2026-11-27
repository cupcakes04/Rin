"""m.dataset and export.yolo on generated media with hand-made boxes (CPU)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from conftest import CFR_STARTS_MS, make_clip

from rin import Master, Run, core, curate, export
from rin.dataset import _without_duplicates


@pytest.fixture
def setup(m: Master, workspace: Path) -> dict[str, Run]:
    """Pool clip P (16 frames, ranked by curate.random) and val clip V (12 frames), one camera;
    a stand-in SAM3 run over both."""
    for name, n in (("P", 16), ("V", 12)):
        folder = workspace / "media" / name
        folder.mkdir()
        make_clip(folder / f"{name}.mkv", CFR_STARTS_MS[:n], rate=10)

    def describe(path: Path) -> dict:
        return {"site": "gym", "camera": "G1", "started_at": datetime(2018, 3, 7, 10 if path.stem == "P" else 11, 0)}

    pool = core.sample(m, core.ingest(m, workspace / "media" / "P", split="pool", describe=describe), fps=10)
    val = core.sample(m, core.ingest(m, workspace / "media" / "V", split="val", describe=describe), fps=10)
    both = curate.union(m, [pool, val])
    keys = both.frame_set().df().sort_values(["media_id", "frame_idx"])[["media_id", "frame_idx"]].values.tolist()
    boxes = []  # (media_id, frame_idx, class, score, x0, y0, x1, y1)
    for media_id, idx in keys:
        if idx % 4 == 3:
            continue  # every fourth frame: nobody there
        boxes.append((media_id, idx, "person", 0.9, 0.1, 0.2, 0.3, 0.6))
        boxes.append((media_id, idx, "person", 0.6, 0.11, 0.2, 0.31, 0.6))   # a near-duplicate of the first
        boxes.append((media_id, idx, "person", 0.3, 0.5, 0.5, 0.52, 0.53))   # weak and tiny
    n = len(boxes)
    with m.writer(step="sam3.concept", config={}, inputs=[both.id]) as w:
        w.write("instances", pa.table({
            "instance_idx": list(range(n)), "media_id": [b[0] for b in boxes],
            "frame_idx": [int(b[1]) for b in boxes], "class": [b[2] for b in boxes], "score": [b[3] for b in boxes],
            **{c: [b[4 + k] for b in boxes] for k, c in enumerate(("x0", "y0", "x1", "y1"))},
            "mask_rle": pa.nulls(n, pa.string()), "keypoints": pa.nulls(n), "track_id": pa.nulls(n, pa.int64()),
        }))
    return {"pool": curate.random(m, pool, n=16, seed=0), "val": val, "labels": m.run(w.run_id), "both": both}


def test_dataset_takes_the_top_frames_and_the_boxes_through_the_gate(m: Master, setup: dict[str, Run]) -> None:
    ds = m.dataset(labels=[setup["labels"]], frames=setup["pool"], top=8, gate="score >= 0.5")
    assert ds.frames == [tuple(r) for r in setup["pool"].frame_set().df().sort_values("rank")[["media_id", "frame_idx"]].values.tolist()[:8]]
    with_people = [k for k in ds.frames if k[1] % 4 != 3]
    assert len(ds.boxes) == 2 * len(with_people)  # the 0.3 box is gated out; both others stay
    assert ds.config == {"frames": setup["pool"].id, "labels": [setup["labels"].id], "top": 8,
                         "gate": "score >= 0.5", "duplicate_iou": None}


def test_dataset_gate_can_use_derived_size_and_drop_duplicates(m: Master, setup: dict[str, Run]) -> None:
    ds = m.dataset(labels=[setup["labels"]], frames=setup["pool"], gate="area > 0.001", duplicate_iou=0.7)
    assert set(ds.boxes.score) == {0.9}  # the tiny box fails the area gate, the 0.6 duplicate is dropped
    assert ds.dropped_duplicates == len(ds.boxes)


def test_duplicates_are_only_boxes_of_one_class() -> None:
    boxes = pd.DataFrame({
        "media_id": ["a"] * 3, "frame_idx": [0] * 3, "class": ["person", "backpack", "person"],
        "score": [0.9, 0.8, 0.7], "x0": [0.1] * 3, "y0": [0.1] * 3, "x1": [0.3] * 3, "y1": [0.5] * 3,
    })
    assert _without_duplicates(boxes, 0.7).tolist() == [True, True, False]


def test_dataset_refuses_frames_no_label_run_saw(m: Master, setup: dict[str, Run], workspace: Path) -> None:
    folder = workspace / "media" / "X"
    folder.mkdir()
    make_clip(folder / "X.mkv", CFR_STARTS_MS[:5], rate=10)
    other = core.sample(m, core.ingest(m, folder, split="pool"), fps=10)
    with pytest.raises(ValueError, match="never given to a label run"):
        m.dataset(labels=[setup["labels"]], frames=other, gate="score >= 0.5")


def test_dataset_refuses_top_on_an_unranked_frame_set(m: Master, setup: dict[str, Run]) -> None:
    with pytest.raises(ValueError, match="ranked"):
        m.dataset(labels=[setup["labels"]], frames=setup["both"], top=3, gate="score >= 0.5")


def test_export_writes_yolo_files_and_freezes_the_dataset(m: Master, setup: dict[str, Run]) -> None:
    train = m.dataset(labels=[setup["labels"]], frames=setup["pool"], top=8, gate="score >= 0.5", duplicate_iou=0.7)
    val = m.dataset(labels=[setup["labels"]], frames=setup["val"], gate="score >= 0.5", duplicate_iou=0.7)
    run = export.yolo(m, train, classes=["person"], val=val, tags={"n": 8})
    files = m.layout.run_dir(run.id) / "files"

    assert (files / "data.yaml").read_text() == 'train: images/train\nval: images/val\nnames:\n  0: "person"\n'
    for split, d in (("train", train), ("val", val)):
        for media_id, idx in d.frames:
            image, label = files / "images" / split / f"{media_id}_{idx:06d}.jpg", files / "labels" / split / f"{media_id}_{idx:06d}.txt"
            assert image.stat().st_ino == m.layout.frame_file(media_id, idx).stat().st_ino  # a hard link
            text = label.read_text()
            if idx % 4 == 3:
                assert text == ""  # nobody there: an empty file, a negative example
            else:
                assert text == "0 0.200000 0.400000 0.200000 0.400000\n"  # centre and size of (0.1, 0.2, 0.3, 0.6)
    assert len(run) == 8 + 12 and len(run.instances().df()) == len(train.boxes) + len(val.boxes)
    assert run.table("artifacts").fetchall() == [("data_yaml", "files/data.yaml")]
    assert run.tags == {"n": 8} and set(run.inputs) == {setup["pool"].id, setup["val"].id, setup["labels"].id}


@pytest.mark.parametrize(("train_from", "val_from", "error"), [
    ("pool", "pool", "val dataset has frames from \\['pool'\\]"),
    ("val", "val", "train dataset has frames from \\['val'\\]"),
])
def test_export_takes_train_from_pool_and_val_from_val(
    m: Master, setup: dict[str, Run], train_from: str, val_from: str, error: str
) -> None:
    train = m.dataset(labels=[setup["labels"]], frames=setup[train_from], gate="score >= 0.5")
    val = m.dataset(labels=[setup["labels"]], frames=setup[val_from], gate="score >= 0.5")
    with pytest.raises(ValueError, match=error):
        export.yolo(m, train, classes=["person"], val=val)


def test_export_refuses_test_frames_in_training(m: Master, setup: dict[str, Run], workspace: Path) -> None:
    folder = workspace / "media" / "T"
    folder.mkdir()
    make_clip(folder / "T.mkv", CFR_STARTS_MS[:4], rate=10)
    test = core.sample(m, core.ingest(m, folder, split="test", describe=lambda _: {
        "site": "gym", "camera": "G1", "started_at": datetime(2018, 3, 9, 10, 0)}), fps=10)
    keys = test.frame_set().df()[["media_id", "frame_idx"]].values.tolist()
    with m.writer(step="sam3.concept", config={}, inputs=[test.id]) as w:
        w.write("instances", pa.table({
            "instance_idx": list(range(len(keys))), "media_id": [k[0] for k in keys],
            "frame_idx": [int(k[1]) for k in keys], "class": ["person"] * len(keys), "score": [0.9] * len(keys),
            "x0": [0.1] * len(keys), "y0": [0.1] * len(keys), "x1": [0.2] * len(keys), "y1": [0.3] * len(keys),
            "mask_rle": pa.nulls(len(keys), pa.string()), "keypoints": pa.nulls(len(keys)),
            "track_id": pa.nulls(len(keys), pa.int64()),
        }))
    both = curate.union(m, [setup["pool"], test])
    train = m.dataset(labels=[setup["labels"], m.run(w.run_id)], frames=both, gate="score >= 0.5")
    val = m.dataset(labels=[setup["labels"]], frames=setup["val"], gate="score >= 0.5")
    with pytest.raises(ValueError, match="train dataset has frames from \\['test'\\]"):
        export.yolo(m, train, classes=["person"], val=val)


def test_export_refuses_a_class_it_was_not_given(m: Master, setup: dict[str, Run]) -> None:
    train = m.dataset(labels=[setup["labels"]], frames=setup["pool"], gate="score >= 0.5")
    val = m.dataset(labels=[setup["labels"]], frames=setup["val"], gate="score >= 0.5")
    with pytest.raises(ValueError, match="not in"):
        export.yolo(m, train, classes=["car"], val=val)
    assert not any(m.layout.tmp.iterdir())
