"""human.suspects on generated media: boxes that stay put, duplicates, split masks (CPU)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pycocotools import mask as mask_util

from rin import Master, Run, core, human

from conftest import CFR_STARTS_MS, make_clip


def _fixed_camera(_path: Path) -> dict:
    return {"site": "gym", "camera": "G1", "started_at": datetime(2018, 3, 13, 15, 55)}


@pytest.fixture
def frames(m: Master, workspace: Path) -> Run:
    folder = workspace / "media" / "cam"
    folder.mkdir()
    make_clip(folder / "cam.mkv", CFR_STARTS_MS, rate=10)
    return core.sample(m, core.ingest(m, folder, split="test", describe=_fixed_camera), fps=10)


def _run(m: Master, frames: Run, boxes: list[tuple], *, config: dict | None = None, masks: list[str] | None = None) -> Run:
    """A stand-in for a labelling run: boxes as (frame rank, x0, y0, x1, y1)."""
    keys = frames.frame_set().df().sort_values("rank")[["media_id", "frame_idx"]].values.tolist()
    n = len(boxes)
    with m.writer(step="sam3.concept", config=config or {}, inputs=[frames.id]) as w:
        w.write("instances", pa.table({
            "instance_idx": list(range(n)), "media_id": [keys[b[0]][0] for b in boxes],
            "frame_idx": [int(keys[b[0]][1]) for b in boxes], "class": ["person"] * n, "score": [0.9] * n,
            **{c: [b[k + 1] for b in boxes] for k, c in enumerate(("x0", "y0", "x1", "y1"))},
            "mask_rle": pa.array(masks if masks is not None else [None] * n, pa.string()),
            "keypoints": pa.nulls(n), "track_id": pa.nulls(n, pa.int64()),
        }))
    return m.run(w.run_id)


@pytest.mark.parametrize("chunk", [256, 1, 3])
def test_a_box_on_the_same_spot_in_most_frames_is_flagged(
    m: Master, frames: Run, monkeypatch: pytest.MonkeyPatch, chunk: int
) -> None:
    """Also in tiny chunks: the check runs over chunks of boxes so a pool hour fits in memory."""
    monkeypatch.setattr(sys.modules["rin.human.suspects"], "SAME_SPOT_CHUNK", chunk)
    n = len(frames)
    bag = [(k, 0.1, 0.6, 0.15, 0.7) for k in range(n)]  # never moves
    walker = [(k, 0.2 + 0.7 * k / n, 0.3, 0.25 + 0.7 * k / n, 0.6) for k in range(n)]  # crosses the room
    flags = human.suspects(m, frames, _run(m, frames, bag + walker))
    assert len(flags) == n and {f["x0"] for f in flags} == {0.1}
    assert flags[0]["reason"] == f"same spot on {n} of {n} frames"


def test_near_duplicate_boxes_on_one_frame_are_flagged(m: Master, frames: Run) -> None:
    pair = [(0, 0.5, 0.5, 0.6, 0.8), (0, 0.5, 0.5, 0.61, 0.8)]
    apart = [(1, 0.1, 0.1, 0.2, 0.4), (1, 0.6, 0.1, 0.7, 0.4)]
    flags = human.suspects(m, frames, _run(m, frames, pair + apart))
    assert len(flags) == 2 and all(f["reason"].startswith("duplicate") for f in flags)


def test_a_box_around_a_split_mask_is_flagged_with_the_detector_box(m: Master, frames: Run) -> None:
    width, height = m.sql("SELECT width, height FROM media").fetchall()[0]
    two_people = np.zeros((height, width), np.uint8)
    two_people[10:40, 5:15] = 1
    two_people[10:40, 50:60] = 1  # a stray piece 35 px away
    one_person = np.zeros((height, width), np.uint8)
    one_person[10:40, 5:15] = 1

    def rle(mask: np.ndarray) -> str:
        return mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")

    def unit(x0: int, y0: int, x1: int, y1: int) -> tuple:
        return (x0 / width, y0 / height, x1 / width, y1 / height)

    checked = _run(m, frames, [(0, *unit(5, 10, 60, 40)), (0, *unit(5, 10, 15, 40))])
    sam = _run(m, frames, [(0, *unit(6, 11, 15, 40)), (0, *unit(5, 10, 15, 40))],
               config={"box": "detector"}, masks=[rle(two_people), rle(one_person)])
    flags = human.suspects(m, frames, checked, masks=sam)
    split = [f for f in flags if f["reason"].startswith("split mask")]
    assert len(split) == 1 and split[0]["reason"] == "split mask (35 px gap)"
    assert split[0]["x1"] == pytest.approx(60 / width)
    assert split[0]["suggested"] == pytest.approx(list(unit(6, 11, 15, 40)))


def test_suspects_needs_a_frame_set_and_runs_with_boxes(m: Master, frames: Run) -> None:
    with pytest.raises(TypeError, match="frame set"):
        human.suspects(m, _run(m, frames, [(0, 0.1, 0.1, 0.2, 0.2)]), frames)
    with pytest.raises(TypeError, match="boxes"):
        human.suspects(m, frames, frames)


def test_suspects_can_look_only_above_a_score(m: Master, frames: Run) -> None:
    pair = [(0, 0.5, 0.5, 0.6, 0.8), (0, 0.5, 0.5, 0.61, 0.8)]
    run = _run(m, frames, pair)
    assert len(human.suspects(m, frames, run, above=0.5)) == 2
    assert human.suspects(m, frames, run, above=0.95) == []
