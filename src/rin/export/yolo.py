"""export.yolo: a train and a val dataset in Ultralytics' YOLO detection format, as a run.

    runs/<id>/files/
        data.yaml                 train: images/train, val: images/val, names; no `path` key,
                                  so Ultralytics resolves the folders beside the file
        images/{train,val}/<media_id>_<frame_idx>.jpg   hard links to the frame cache
        labels/{train,val}/<media_id>_<frame_idx>.txt   "class xc yc w h" per box, in [0, 1];
                                                        empty for a frame with nobody on it

Hard links, not copies or symlinks: no extra disk, and the export still works after the frame
cache (derived data, D-29) is deleted. They need the store and its cache on one file system.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pyarrow as pa

from rin.dataset import Dataset
from rin.master import Master
from rin.run import Run
from rin.servants.frame_cache import cache_frames


def yolo(m: Master, ds: Dataset, *, classes: list[str], val: Dataset, tags: dict[str, Any] | None = None) -> Run:
    """Write `ds` as the train split and `val` as the val split. `classes` lists the class names
    in YOLO's order (index 0 first); a box of any other class is refused. No recording may be
    in both splits (D-13)."""
    if not isinstance(classes, list) or not classes or any(not isinstance(c, str) or not c for c in classes):
        raise ValueError(f"classes must be a non-empty list of class names, got {classes!r}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"classes lists a name twice: {classes}")
    for name, d in (("train", ds), ("val", val)):
        if not isinstance(d, Dataset):
            raise TypeError(f"{name} must be an m.dataset(...), got {type(d).__name__}")
        if not len(d):
            raise ValueError(f"The {name} dataset has no frames")
        other = sorted(set(d.boxes["class"]) - set(classes))
        if other:
            raise ValueError(f"The {name} dataset has boxes of classes {other}, not in {classes}")
    shared = {k[0] for k in ds.frames} & {k[0] for k in val.frames}
    if shared:
        raise ValueError(f"{len(shared)} recordings are in both train and val (D-13), e.g. {sorted(shared)[:3]}")

    index = {c: i for i, c in enumerate(classes)}
    inputs = list(dict.fromkeys([*ds.inputs, *val.inputs]))
    config = {"classes": classes, "train": ds.config, "val": val.config}
    with m.writer(step="export.yolo", config=config, inputs=inputs, tags=tags) as w:
        files = w.tmp_dir / "files"
        frame_rows: dict[str, list] = {"media_id": [], "frame_idx": [], "rank": []}
        exported = []
        for split, d in (("train", ds), ("val", val)):
            images, labels = files / "images" / split, files / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            cached = {(media_id, idx): path for media_id, idx, path in cache_frames(m, d.frames)}
            by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
            for b in d.boxes.to_dict("records"):
                by_frame.setdefault((b["media_id"], int(b["frame_idx"])), []).append(b)
            for rank, key in enumerate(d.frames):
                name = f"{key[0]}_{key[1]:06d}"
                os.link(cached[key], images / f"{name}.jpg")
                lines = []
                for b in by_frame.get((key[0], int(key[1])), []):
                    lines.append(
                        f"{index[b['class']]} {(b['x0'] + b['x1']) / 2:.6f} {(b['y0'] + b['y1']) / 2:.6f} "
                        f"{b['x1'] - b['x0']:.6f} {b['y1'] - b['y0']:.6f}\n"
                    )
                    exported.append(b)
                (labels / f"{name}.txt").write_text("".join(lines))
                frame_rows["media_id"].append(key[0])
                frame_rows["frame_idx"].append(int(key[1]))
                frame_rows["rank"].append(rank if split == "train" else None)
        names = "".join(f"  {i}: {json.dumps(c)}\n" for i, c in enumerate(classes))
        (files / "data.yaml").write_text(f"train: images/train\nval: images/val\nnames:\n{names}")

        n = len(exported)
        w.write("frame_sets", pa.table({
            "media_id": frame_rows["media_id"],
            "frame_idx": pa.array(frame_rows["frame_idx"], pa.int64()),
            "rank": pa.array(frame_rows["rank"], pa.int64()),
        }))
        w.write("instances", pa.table({
            "instance_idx": pa.array(range(n), pa.int64()),
            "media_id": [b["media_id"] for b in exported],
            "frame_idx": pa.array([int(b["frame_idx"]) for b in exported], pa.int64()),
            "class": [b["class"] for b in exported],
            # NaN is pandas' null: a person's box has no score
            "score": pa.array([None if b["score"] != b["score"] else float(b["score"]) for b in exported], pa.float64()),
            **{c: pa.array([float(b[c]) for b in exported], pa.float64()) for c in ("x0", "y0", "x1", "y1")},
            "mask_rle": pa.nulls(n, pa.string()),
            "keypoints": pa.nulls(n),
            "track_id": pa.nulls(n, pa.int64()),
        }))
        w.write("artifacts", pa.table({"name": ["data_yaml"], "path": ["files/data.yaml"]}))
        print(f"export.yolo: train {len(ds)} frames, {len(ds.boxes)} boxes; val {len(val)} frames, {len(val.boxes)} boxes")
    return m.run(w.run_id)
