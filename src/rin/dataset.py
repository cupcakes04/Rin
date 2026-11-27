"""A training set as a query (D-02): which frames, and which of the label runs' boxes on them.

`m.dataset(...)` writes nothing. An export of it is the run that freezes it (`export.yolo`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rin.run import Run

if TYPE_CHECKING:
    from rin.master import Master

DERIVED = "(x1 - x0) AS w, (y1 - y0) AS h, (x1 - x0) * (y1 - y0) AS area"
"""Columns a gate may use besides the instances table's own; [0, 1] units, like the boxes."""


class Dataset:
    """Frames and boxes for training, with the definition that picked them.

    frames:  the frame set's frames, the first `top` by rank when `top` is given, as a list of
             (media_id, frame_idx) in rank order (or media and frame order when unranked).
    boxes:   a DataFrame of the label runs' boxes on those frames that pass the gate (and the
             duplicate rule): media_id, frame_idx, class, score, x0, y0, x1, y1, run_id, instance_idx.
    config:  the definition, for the run that exports it.
    inputs:  the runs it reads.
    """

    def __init__(
        self,
        m: Master,
        *,
        labels: list[Run],
        frames: Run,
        top: int | None,
        gate: str,
        duplicate_iou: float | None,
    ) -> None:
        if not labels or any("instances" not in r.tables for r in labels):
            raise TypeError(f"labels must be runs with boxes; got {[r.id for r in labels]}")
        if len({r.id for r in labels}) != len(labels):
            raise ValueError("labels lists a run twice")
        if "frame_sets" not in frames.tables:
            raise TypeError(f"frames must be a frame set; {frames.id} wrote {frames.tables}")
        if top is not None and (not isinstance(top, int) or isinstance(top, bool) or top < 1):
            raise ValueError(f"top must be a positive integer or None, got {top!r}")
        if not isinstance(gate, str) or not gate.strip():
            raise ValueError("gate must be an SQL condition, e.g. 'score >= 0.5'")
        if duplicate_iou is not None and not (
            isinstance(duplicate_iou, (int, float)) and not isinstance(duplicate_iou, bool) and 0 < duplicate_iou <= 1
        ):
            raise ValueError(f"duplicate_iou must be in (0, 1] or None, got {duplicate_iou!r}")

        rows = m.sql("SELECT media_id, frame_idx, rank FROM frame_sets WHERE run_id = ?", [frames.id]).fetchall()
        if top is not None:
            if any(rank is None for _, _, rank in rows):
                raise ValueError(f"top needs a ranked frame set; {frames.id} has frames without a rank")
            if top > len(rows):
                raise ValueError(f"top={top} but {frames.id} holds {len(rows)} frames")
            rows = sorted(rows, key=lambda r: r[2])[:top]
        else:
            rows = sorted(rows, key=lambda r: (r[2] is None, r[2], r[0], r[1]))
        self.frames: list[tuple[str, int]] = [(media_id, idx) for media_id, idx, _ in rows]

        label_ids = [r.id for r in labels]
        _check_labelled(m, labels, self.frames)
        boxes = m.sql(
            f"""SELECT media_id, frame_idx, class, score, x0, y0, x1, y1, run_id, instance_idx
                FROM (SELECT *, {DERIVED} FROM instances WHERE list_contains(?, run_id))
                WHERE ({gate})
                ORDER BY media_id, frame_idx, run_id, instance_idx""",
            [label_ids],
        ).df()
        wanted = set(self.frames)
        boxes = boxes[[k in wanted for k in zip(boxes.media_id, boxes.frame_idx)]].reset_index(drop=True)
        self.dropped_duplicates = 0
        if duplicate_iou is not None and len(boxes):
            keep = _without_duplicates(boxes, duplicate_iou)
            self.dropped_duplicates = int((~keep).sum())
            boxes = boxes[keep].reset_index(drop=True)
        self.boxes: pd.DataFrame = boxes
        self.config: dict[str, Any] = {
            "frames": frames.id, "labels": label_ids, "top": top, "gate": gate, "duplicate_iou": duplicate_iou,
        }
        self.inputs: list[str] = [frames.id, *label_ids]

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return f"Dataset({len(self.frames)} frames, {len(self.boxes)} boxes, {self.config})"


def _check_labelled(m: Master, labels: list[Run], frames: list[tuple[str, int]]) -> None:
    """Every frame was given to exactly one label run. A frame no run labelled would export as a
    frame with nobody on it; a frame two runs labelled would get both runs' boxes."""
    covering: list[tuple[str, str]] = []  # (label run, frame set it labelled)
    for r in labels:
        if "frame_sets" in r.tables:  # e.g. human.annotate: its own frame set is what it finished
            covering.append((r.id, r.id))
        else:  # e.g. sam3.concept: the frame sets it was given
            covering += [(r.id, i) for i in r.inputs if "frame_sets" in m.run(i).tables]
    seen: dict[tuple[str, int], set[str]] = {}
    for run_id, frame_set in covering:
        for key in m.sql("SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ?", [frame_set]).fetchall():
            seen.setdefault(key, set()).add(run_id)
    unlabelled = [k for k in frames if k not in seen]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} frames were never given to a label run, e.g. {unlabelled[:3]}")
    twice = [k for k in frames if len(seen[k]) > 1]
    if twice:
        raise ValueError(f"{len(twice)} frames were labelled by more than one label run, e.g. {twice[:3]}")


def _without_duplicates(boxes: pd.DataFrame, iou_min: float) -> np.ndarray:
    """Per frame, highest score first (a person's box, without a score, counts as highest):
    keep a box unless it overlaps a box already kept by at least `iou_min`."""
    keep = np.zeros(len(boxes), bool)
    score = boxes.score.fillna(np.inf).to_numpy()
    xyxy = boxes[["x0", "y0", "x1", "y1"]].to_numpy()
    for _, idx in boxes.groupby(["media_id", "frame_idx"]).indices.items():
        kept: list[int] = []
        for i in idx[np.argsort(-score[idx], kind="stable")]:
            if kept:
                k = xyxy[kept]
                w = np.clip(np.minimum(xyxy[i, 2], k[:, 2]) - np.maximum(xyxy[i, 0], k[:, 0]), 0, None)
                h = np.clip(np.minimum(xyxy[i, 3], k[:, 3]) - np.maximum(xyxy[i, 1], k[:, 1]), 0, None)
                inter = w * h
                area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
                if (inter / (area(xyxy[i]) + area(k) - inter)).max() >= iou_min:
                    continue
            kept.append(i)
        keep[kept] = True
    return keep
