"""evaluate.coco: COCO's box evaluation (pycocotools) of a predictions run against an
export.coco run. Crowd regions (`iscrowd = 1`, D-34) neither reward nor punish a detection.

Metrics (over all classes): mAP50-95, mAP50, mAP75, mAP_small / _medium / _large (COCO's
sizes: under 32² px, up to 96² px, above), AR1, AR10, AR100 and AR by size, and fp_per_empty_frame:
boxes scoring at least 0.5 on frames with no box and no crowd region, per such frame. A size
with no checked box has no mAP; that row is left out rather than stored as COCO's -1.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any

import pyarrow as pa
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rin.master import Master
from rin.run import Run

STATS = ["mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
         "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
"""COCOeval.stats, in its order."""
EMPTY_FRAME_SCORE = 0.5
"""The confidence above which a box on an empty frame counts as a false positive."""


def coco(m: Master, predictions: Run, truth: Run, *, tags: dict[str, Any] | None = None) -> Run:
    """Score `predictions` (an instances run, e.g. yolo.predict, made on every truth frame)
    against `truth` (an export.coco run)."""
    if truth.step != "export.coco":
        raise TypeError(f"truth must be an export.coco run; {truth.id} is {truth.step}")
    if "instances" not in predictions.tables:
        raise TypeError(f"predictions must be a run with boxes; {predictions.id} wrote {predictions.tables}")
    doc = json.loads((m.layout.run_dir(truth.id) / "files" / "annotations.json").read_text())
    image_of = {(im["rin_media_id"], im["rin_frame_idx"]): im for im in doc["images"]}
    category = {c["name"]: c["id"] for c in doc["categories"]}

    predicted = set()
    for i in predictions.inputs:
        if "frame_sets" in m.run(i).tables:
            predicted |= set(m.sql("SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ?", [i]).fetchall())
    unseen = [k for k in image_of if k not in predicted]
    if unseen:
        raise ValueError(f"{len(unseen)} truth frames were not given to {predictions.id}, e.g. {unseen[:3]}")
    rows = m.sql(
        "SELECT media_id, frame_idx, class, score, x0, y0, x1, y1 FROM instances WHERE run_id = ?", [predictions.id]
    ).fetchall()
    other = sorted({r[2] for r in rows} - set(category))
    if other:
        raise ValueError(f"{predictions.id} has boxes of classes {other}; truth knows {sorted(category)}")
    if any(r[3] is None for r in rows):
        raise ValueError(f"{predictions.id} has boxes without a score; COCO's evaluation ranks by score")
    dets = []
    for media_id, idx, cls, score, x0, y0, x1, y1 in rows:
        im = image_of.get((media_id, idx))
        if im is None:
            continue  # predicted on a frame the truth doesn't hold
        w, h = im["width"], im["height"]
        dets.append({"image_id": im["id"], "category_id": category[cls], "score": score,
                     "bbox": [x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h]})
    if not dets:
        raise ValueError(f"{predictions.id} has no boxes on the truth frames; nothing to score")

    gt = COCO()
    gt.dataset = doc
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    values = {name: float(v) for name, v in zip(STATS, ev.stats, strict=True) if v >= 0}

    empty = {im["id"] for im in doc["images"]} - {a["image_id"] for a in doc["annotations"]}
    if empty:
        wrong = sum(d["score"] >= EMPTY_FRAME_SCORE and d["image_id"] in empty for d in dets)
        values["fp_per_empty_frame"] = wrong / len(empty)

    config = {"empty_frame_score": EMPTY_FRAME_SCORE, "max_dets": 100}
    with m.writer(step="evaluate.coco", config=config, inputs=[predictions.id, truth.id], tags=tags) as w:
        w.write("metrics", pa.table({
            "name": list(values), "class": pa.nulls(len(values), pa.string()),
            "value": pa.array(list(values.values()), pa.float64()),
        }))
        print(printed.getvalue(), end="")
        print(f"evaluate.coco: mAP50-95 {values['mAP50-95']:.3f}, mAP50 {values['mAP50']:.3f}"
              + (f", {values['fp_per_empty_frame']:.2f} false positives per empty frame" if "fp_per_empty_frame" in values else ""))
    return m.run(w.run_id)
