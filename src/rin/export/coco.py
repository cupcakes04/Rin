"""export.coco: a checked run as COCO ground truth (`files/annotations.json`), as a run.

Every frame of the run's frame set becomes an image, empty ones included (where false positives
show up, D-30). Boxes go out in pixels as [x, y, w, h] with `area` = w * h. A crowd box (a region
where a person couldn't tell people apart, D-34) goes out with `iscrowd = 1` under the class it
stands for, so COCO's evaluation neither rewards nor punishes a detection inside it.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from rin.master import Master
from rin.run import Run


def coco(m: Master, truth: Run, *, classes: list[str], crowd: dict[str, str] | None = None,
         tags: dict[str, Any] | None = None) -> Run:
    """Write `truth` (a run with a frame set of complete frames and their boxes, e.g. a
    `human.annotate` run) as COCO ground truth. `classes` become categories 1, 2, ...;
    `crowd` maps a crowd class to the class it covers, e.g. {"crowd": "person"}. A box of any
    other class is refused."""
    if not isinstance(classes, list) or not classes or any(not isinstance(c, str) or not c for c in classes):
        raise ValueError(f"classes must be a non-empty list of class names, got {classes!r}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"classes lists a name twice: {classes}")
    crowd = dict(crowd or {})
    if any(k in classes or v not in classes for k, v in crowd.items()):
        raise ValueError(f"crowd must map crowd classes (not in classes) to classes in {classes}, got {crowd}")
    if "frame_sets" not in truth.tables or "instances" not in truth.tables:
        raise TypeError(f"truth must be a run with a frame set and boxes; {truth.id} wrote {truth.tables}")

    frames = m.sql(
        """SELECT s.media_id, s.frame_idx, d.width, d.height FROM frame_sets s JOIN media d USING (media_id)
           WHERE s.run_id = ? ORDER BY s.media_id, s.frame_idx""",
        [truth.id],
    ).fetchall()
    boxes = m.sql(
        "SELECT media_id, frame_idx, class, x0, y0, x1, y1 FROM instances WHERE run_id = ? ORDER BY media_id, frame_idx, instance_idx",
        [truth.id],
    ).fetchall()
    other = sorted({b[2] for b in boxes} - set(classes) - set(crowd))
    if other:
        raise ValueError(f"{truth.id} has boxes of classes {other}, neither in classes nor crowd")
    image_id = {(media_id, idx): i + 1 for i, (media_id, idx, _, _) in enumerate(frames)}
    size = {(media_id, idx): (w, h) for media_id, idx, w, h in frames}
    outside = [b[:2] for b in boxes if (b[0], b[1]) not in image_id]
    if outside:
        raise ValueError(f"{len(outside)} boxes sit on frames outside {truth.id}'s frame set, e.g. {outside[:3]}")
    category = {c: i + 1 for i, c in enumerate(classes)}

    annotations = []
    for media_id, idx, cls, x0, y0, x1, y1 in boxes:
        w, h = size[(media_id, idx)]
        bw, bh = (x1 - x0) * w, (y1 - y0) * h
        annotations.append({
            "id": len(annotations) + 1, "image_id": image_id[(media_id, idx)],
            "category_id": category[crowd.get(cls, cls)], "bbox": [x0 * w, y0 * h, bw, bh],
            "area": bw * bh, "iscrowd": int(cls in crowd),
        })
    doc = {
        "images": [
            {"id": image_id[(media_id, idx)], "file_name": f"{media_id}_{idx:06d}.jpg", "width": w, "height": h,
             "rin_media_id": media_id, "rin_frame_idx": idx}
            for media_id, idx, w, h in frames
        ],
        "annotations": annotations,
        "categories": [{"id": i, "name": c} for c, i in category.items()],
    }
    config = {"classes": classes, "crowd": crowd}
    with m.writer(step="export.coco", config=config, inputs=[truth.id], tags=tags) as w:
        (w.tmp_dir / "files").mkdir()
        (w.tmp_dir / "files" / "annotations.json").write_text(json.dumps(doc))
        w.write("artifacts", pa.table({"name": ["annotations"], "path": ["files/annotations.json"]}))
        n_crowd = sum(a["iscrowd"] for a in annotations)
        print(f"export.coco: {len(frames)} images, {len(annotations) - n_crowd} boxes, {n_crowd} crowd regions")
    return m.run(w.run_id)
