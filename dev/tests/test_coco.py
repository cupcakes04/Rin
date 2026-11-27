"""export.coco and evaluate.coco on hand-made truth and predictions (CPU)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pytest
from conftest import CFR_STARTS_MS, make_clip

from rin import Master, Run, core, evaluate, export

PERSON = (0.10, 0.20, 0.30, 0.60)
CROWD = (0.50, 0.10, 0.90, 0.50)


def _boxes(m: Master, step: str, inputs: list[str], boxes: list[tuple], *, frame_set: Run | None = None) -> Run:
    """A run of boxes (media_id, frame_idx, class, score, x0, y0, x1, y1); with `frame_set`,
    it also writes that frame set as its own, like human.annotate."""
    n = len(boxes)
    with m.writer(step=step, config={}, inputs=inputs) as w:
        if frame_set is not None:
            w.write("frame_sets", frame_set.frame_set().to_arrow_table())
        w.write("instances", pa.table({
            "instance_idx": pa.array(range(n), pa.int64()), "media_id": [b[0] for b in boxes],
            "frame_idx": pa.array([b[1] for b in boxes], pa.int64()), "class": [b[2] for b in boxes],
            "score": pa.array([b[3] for b in boxes], pa.float64()),
            **{c: pa.array([b[4 + k] for b in boxes], pa.float64()) for k, c in enumerate(("x0", "y0", "x1", "y1"))},
            "mask_rle": pa.nulls(n, pa.string()), "keypoints": pa.nulls(n), "track_id": pa.nulls(n, pa.int64()),
        }))
    return m.run(w.run_id)


@pytest.fixture
def scene(m: Master, workspace: Path) -> dict:
    """4 test frames: frame 0 one person; frame 1 one person and a crowd region; frames 2, 3 empty."""
    folder = workspace / "media" / "T"
    folder.mkdir()
    make_clip(folder / "T.mkv", CFR_STARTS_MS[:4], rate=10)
    frames = core.sample(m, core.ingest(m, folder, split="test",
                                        describe=lambda p: {"camera": "G1", "started_at": datetime(2018, 3, 9)}), fps=10)
    mid = frames.frame_set().df().media_id[0]
    truth = _boxes(m, "human.annotate", [frames.id], [
        (mid, 0, "person", None, *PERSON), (mid, 1, "person", None, *PERSON), (mid, 1, "crowd", None, *CROWD),
    ], frame_set=frames)
    return {"frames": frames, "truth": truth, "mid": mid}


def test_export_coco_writes_pixels_crowds_and_empty_frames(m: Master, scene: dict) -> None:
    run = export.coco(m, scene["truth"], classes=["person"], crowd={"crowd": "person"})
    doc = json.loads((m.layout.run_dir(run.id) / "files" / "annotations.json").read_text())
    assert len(doc["images"]) == 4 and doc["categories"] == [{"id": 1, "name": "person"}]
    w, h = doc["images"][0]["width"], doc["images"][0]["height"]
    assert doc["annotations"][0]["bbox"] == pytest.approx([0.1 * w, 0.2 * h, 0.2 * w, 0.4 * h])
    assert [a["iscrowd"] for a in doc["annotations"]] == [0, 0, 1]
    assert {a["category_id"] for a in doc["annotations"]} == {1}


def test_export_coco_refuses_an_unknown_class(m: Master, scene: dict) -> None:
    with pytest.raises(ValueError, match="crowd"):
        export.coco(m, scene["truth"], classes=["person"])


def test_export_coco_refuses_a_class_listed_twice(m: Master, scene: dict) -> None:
    with pytest.raises(ValueError, match="twice"):
        export.coco(m, scene["truth"], classes=["person", "person"], crowd={"crowd": "person"})


@pytest.mark.parametrize(
    ("extra", "map_", "fp"),
    [
        ([], 1.0, 0.0),                                              # exactly the truth
        ([(1, 0.9, 0.55, 0.15, 0.65, 0.35)], 1.0, 0.0),              # inside the crowd: ignored
        ([(2, 0.99, 0.1, 0.1, 0.2, 0.2), (3, 0.3, 0.1, 0.1, 0.2, 0.2)], None, 0.5),  # empty frames: one above 0.5, ranked above the truth
    ],
    ids=["perfect", "crowd ignored", "false positives"],
)
def test_evaluate_coco_scores_against_checked_truth(m: Master, scene: dict, extra: list, map_: float | None, fp: float) -> None:
    truth = export.coco(m, scene["truth"], classes=["person"], crowd={"crowd": "person"})
    mid = scene["mid"]
    preds = _boxes(m, "yolo.predict", [scene["frames"].id], [
        (mid, 0, "person", 0.95, *PERSON), (mid, 1, "person", 0.95, *PERSON),
        *[(mid, f, "person", s, *box) for f, s, *box in extra],
    ])
    run = evaluate.coco(m, preds, truth)
    metrics = dict(run.table("metrics").df()[["name", "value"]].values)
    if map_ is not None:
        assert metrics["mAP50-95"] == pytest.approx(map_)
    else:
        assert metrics["mAP50-95"] < 1.0
    assert metrics["fp_per_empty_frame"] == pytest.approx(fp)
    assert run.inputs == [preds.id, truth.id]


def test_evaluate_coco_refuses_predictions_that_missed_truth_frames(m: Master, scene: dict) -> None:
    truth = export.coco(m, scene["truth"], classes=["person"], crowd={"crowd": "person"})
    preds = _boxes(m, "yolo.predict", [], [(scene["mid"], 0, "person", 0.9, *PERSON)])
    with pytest.raises(ValueError, match="not given to"):
        evaluate.coco(m, preds, truth)
