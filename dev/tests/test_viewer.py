"""The viewer's caller and human.annotate, on generated media (CPU; the FiftyOne job is faked)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rin import Master, Run, core, curate, human, viewer

VIEWER_MODULE = sys.modules["rin.viewer"]


@pytest.fixture
def frames(m: Master, clips: Path) -> Run:
    return curate.random(m, core.sample(m, core.ingest(m, clips, split="test"), fps=4), n=6, seed=0)


def _boxes_run(m: Master, frames: Run) -> Run:
    """A stand-in for SAM3: one box per frame, the left half of the image."""
    import pyarrow as pa

    rows = frames.frame_set().df()
    n = len(rows)
    with m.writer(step="sam3.concept", config={}, inputs=[frames.id]) as w:
        w.write("instances", pa.table({
            "instance_idx": list(range(n)), "media_id": rows["media_id"].tolist(),
            "frame_idx": rows["frame_idx"].tolist(), "class": ["person"] * n, "score": [0.8] * n,
            "x0": [0.0] * n, "y0": [0.0] * n, "x1": [0.5] * n, "y1": [1.0] * n,
            "mask_rle": pa.nulls(n, pa.string()), "keypoints": pa.nulls(n), "track_id": pa.nulls(n, pa.int64()),
        }))
    return m.run(w.run_id)


def test_show_sends_frames_in_rank_order_with_each_runs_boxes(
    m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: dict = {}

    def capture(_job_dir: Path, spec_file: Path) -> None:
        sent.update(json.loads(spec_file.read_text()))

    monkeypatch.setattr(VIEWER_MODULE, "run_job", capture)
    labels = _boxes_run(m, frames)
    viewer.show(m, frames, name="pilot", labels=[labels], draw="drawn")

    ranked = frames.frame_set().df().sort_values("rank")[["media_id", "frame_idx"]].values.tolist()
    assert [[f["media_id"], f["frame_idx"]] for f in sent["frames"]] == ranked
    field = f"run_sam3_concept_{labels.id[-4:]}"
    assert sent["label_fields"] == [field] and sent["draw_field"] == "drawn" and sent["classes"] == ["person"]
    first = sent["frames"][0]
    assert first["labels"][field] == [{"class": "person", "score": 0.8, "xywh": [0.0, 0.0, 0.5, 1.0]}]
    assert Path(first["path"]).is_file()  # the cached JPEG


@pytest.mark.parametrize(("kwargs", "error"), [({"name": "has space"}, "name"), ({"draw": "run_x"}, "draw"), ({"classes": []}, "classes")])
def test_show_refuses_bad_arguments(m: Master, frames: Run, kwargs: dict, error: str) -> None:
    args = {"name": "pilot", "draw": "drawn", **kwargs}
    with pytest.raises(ValueError, match=error):
        viewer.show(m, frames, **args)


def _fake_dump(frames: Run, boxes_per_frame: list, *, done: object = "all", extra_frame: bool = False, info: dict | None = None):
    """Stands in for the viewer job's dump: the frames of `frames` in rank order, cycling through
    `boxes_per_frame`; `done` = "all" or the ranks tagged done."""
    rows = frames.frame_set().df().sort_values("rank")[["media_id", "frame_idx"]].values.tolist()
    if extra_frame:
        rows = [*rows, ["m-elsewhere", 0]]

    def run(_job_dir: Path, spec_file: Path) -> None:
        spec = json.loads(spec_file.read_text())
        assert spec["command"] == "dump"
        Path(spec["out"]).write_text(json.dumps({
            "fiftyone": "fake", "info": info if info is not None else {"frames": frames.id},
            "frames": [
                {"media_id": mid, "frame_idx": int(idx),
                 "tags": ["done"] if done == "all" or k in done else [],
                 "boxes": boxes_per_frame[k % len(boxes_per_frame)]}
                for k, (mid, idx) in enumerate(rows)
            ],
        }))

    return run


BOX = {"class": "person", "xywh": [0.1, 0.2, 0.3, 0.4]}


def test_annotate_files_finished_frames_and_their_boxes(m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    drawn = [[BOX], [], [{"class": "person", "xywh": [0.5, 0.5, 0.5, 0.5]}]]
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, drawn))
    run = human.annotate(m, frames, view="pilot", field="drawn", who="nic")

    df = run.instances().df()
    assert len(df) == 4  # 6 frames cycling through 1, 0, 1 boxes
    assert len(run) == 6  # the empty finished frames are filed too: "nothing here" is an answer
    assert df["score"].isna().all() and (df["class"] == "person").all()
    first = df.sort_values("x0").iloc[0]
    assert first[["x0", "y0"]].tolist() == pytest.approx([0.1, 0.2]) and first[["x1", "y1"]].tolist() == pytest.approx([0.4, 0.6])
    assert df["x1"].max() == 1.0  # 0.5 + 0.5 lands exactly on the edge
    assert run.info["who"] == "nic" and run.info["code"]["fiftyone"] == "fake" and run.inputs == [frames.id]
    assert run.config == {"view": "pilot", "field": "drawn", "done_tag": "done",
                          "start_from": None, "start_above": None, "label_scores": None}


def test_annotate_files_only_the_frames_tagged_done(m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    drawn = [[BOX], [], [], [], [], []]  # boxes only on rank 0
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, drawn, done={0, 1}))
    run = human.annotate(m, frames, view="pilot", field="drawn", who="nic")
    ranked = frames.frame_set().df().sort_values("rank")[["media_id", "frame_idx"]].values.tolist()
    assert sorted(run.frame_set().df()[["media_id", "frame_idx"]].values.tolist()) == sorted(ranked[:2])
    assert len(run.instances().df()) == 1


def test_annotate_refuses_boxes_on_an_unfinished_frame(m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, [[BOX]], done={0}))
    with pytest.raises(ValueError, match="no 'done' tag"):
        human.annotate(m, frames, view="pilot", field="drawn", who="nic")
    assert not any(m.layout.tmp.iterdir())


def test_annotate_refuses_nothing_done_and_frames_from_elsewhere(m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, [[]], done=set()))
    with pytest.raises(ValueError, match="No frame"):
        human.annotate(m, frames, view="pilot", field="drawn", who="nic")
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, [[]], extra_frame=True))
    with pytest.raises(ValueError, match="not in"):
        human.annotate(m, frames, view="pilot", field="drawn", who="nic")


def test_annotate_clips_a_box_drawn_past_the_edge_but_refuses_one_outside(
    m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    past_edge = [[{"class": "person", "xywh": [-0.0024, 0.5, 0.1, 0.6]}]]  # as drawn in the pilot
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, past_edge))
    run = human.annotate(m, frames, view="pilot", field="drawn", who="nic")
    assert run.instances().df()[["x0", "y1"]].values.tolist()[0] == [0.0, 1.0]

    outside = [[{"class": "person", "xywh": [1.05, 0.0, 0.1, 0.5]}]]  # nothing left inside the image
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, outside))
    with pytest.raises(ValueError, match="outside"):
        human.annotate(m, frames, view="pilot", field="drawn", who="nic")


def test_annotate_needs_a_person(m: Master, frames: Run) -> None:
    with pytest.raises(ValueError, match="who"):
        human.annotate(m, frames, view="pilot", field="drawn", who=" ")


def _scored_run(m: Master, frames: Run, scores: list[float]) -> Run:
    """A stand-in for SAM3 with one box per score on every frame."""
    import pyarrow as pa

    rows = frames.frame_set().df()
    keys = [(r.media_id, int(r.frame_idx), s) for r in rows.itertuples() for s in scores]
    n = len(keys)
    with m.writer(step="sam3.concept", config={}, inputs=[frames.id]) as w:
        w.write("instances", pa.table({
            "instance_idx": list(range(n)), "media_id": [k[0] for k in keys], "frame_idx": [k[1] for k in keys],
            "class": ["person"] * n, "score": [k[2] for k in keys],
            "x0": [0.1] * n, "y0": [0.1] * n, "x1": [0.2 + k[2] / 2 for k in keys], "y1": [0.9] * n,
            "mask_rle": pa.nulls(n, pa.string()), "keypoints": pa.nulls(n), "track_id": pa.nulls(n, pa.int64()),
        }))
    return m.run(w.run_id)


def test_show_starts_from_a_runs_boxes_above_the_gate_and_shows_a_band_as_hints(
    m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: dict = {}
    monkeypatch.setattr(VIEWER_MODULE, "run_job", lambda _d, spec: sent.update(json.loads(spec.read_text())))
    sam = _scored_run(m, frames, [0.2, 0.4, 0.8])
    viewer.show(m, frames, name="check", draw="checked", start_from=sam, start_above=0.5,
                labels=[sam], label_scores=(0.3, 0.5))
    frame = sent["frames"][0]
    assert [b["score"] for b in frame["start"]] == [0.8]
    assert [b["score"] for b in frame["labels"]["hints_sam3_0_3_to_0_5"]] == [0.4]
    assert sent["info"] == {"frames": frames.id, "labels": [sam.id], "label_scores": [0.3, 0.5],
                            "start_from": sam.id, "start_above": 0.5}


def test_show_refuses_start_from_without_a_draw_field(m: Master, frames: Run) -> None:
    sam = _scored_run(m, frames, [0.8])
    with pytest.raises(ValueError, match="draw field"):
        viewer.show(m, frames, name="check", start_from=sam)


def test_annotate_records_the_runs_the_view_showed_and_started_from(
    m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    sam = _scored_run(m, frames, [0.8])
    info = {"frames": frames.id, "labels": [sam.id], "label_scores": [0.3, 0.5], "start_from": sam.id, "start_above": 0.5}
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, [[BOX]], info=info))
    run = human.annotate(m, frames, view="check", field="checked", who="nic")
    assert run.inputs == [frames.id, sam.id]
    assert run.config["start_from"] == sam.id and run.config["start_above"] == 0.5


def test_annotate_refuses_a_view_built_from_another_frame_set(
    m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = curate.random(m, frames, n=3, seed=1)
    monkeypatch.setattr(VIEWER_MODULE, "run_job", _fake_dump(frames, [[]], info={"frames": other.id}))
    with pytest.raises(ValueError, match="was built from"):
        human.annotate(m, frames, view="check", field="checked", who="nic")

