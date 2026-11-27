"""yolo.train and yolo.predict: an Ultralytics YOLO detector, trained on an export.yolo run or
run on a frame set, as batch jobs in the YOLO environment.

The caller (this file) runs in the core environment; the job (job.py) in this folder's own
environment, importing nothing from Rin. They meet in the run's temporary folder:

    caller: copy the export into data/ (hard links), write spec.json  ->  job: train into train/
    caller: move weights and results into files/, record metrics, file the run

Training reads a copy because Ultralytics writes a labels cache beside the labels it reads, and
the export is a finished run that nothing may change (D-23).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa

from rin.master import Master
from rin.run import Run
from rin.servants.checksum import file_sha256
from rin.servants.frame_cache import cached_frames
from rin.servants.launch import run_job

JOB_DIR = Path(__file__).resolve().parent
METRICS = {
    "metrics/precision(B)": "val/precision",
    "metrics/recall(B)": "val/recall",
    "metrics/mAP50(B)": "val/mAP50",
    "metrics/mAP50-95(B)": "val/mAP50-95",
}
"""Ultralytics' names for best.pt's val scores -> the names stored in `metrics`. Val holds SAM3's
unchecked labels: these numbers pick a checkpoint, they don't score the student (D-31)."""


def train(
    m: Master,
    export: Run,
    *,
    model: str | Path,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int,
    workers: int = 8,
    tags: dict[str, Any] | None = None,
) -> Run:
    """Train `model` (a stock checkpoint file, e.g. workspace/checkpoints/yolo/yolo26s.pt) on the
    train split of `export` for `epochs` epochs, keeping as best.pt the epoch with the best val
    score (no early stop). `imgsz` is the training image size (a multiple of 32); `batch`
    images per step; `seed` seeds Ultralytics (deterministic mode)."""
    if export.step != "export.yolo":
        raise TypeError(f"train needs an export.yolo run; {export.id} is {export.step}")
    model = Path(model).resolve()
    if not model.is_file() or model.suffix != ".pt":
        raise FileNotFoundError(f"No YOLO checkpoint (.pt) at {model}")
    for name, value, low in (("epochs", epochs, 1), ("imgsz", imgsz, 32), ("batch", batch, 1),
                             ("seed", seed, 0), ("workers", workers, 0)):
        if not isinstance(value, int) or isinstance(value, bool) or value < low:
            raise ValueError(f"{name} must be an integer of at least {low}, got {value!r}")
    if imgsz % 32:
        raise ValueError(f"imgsz must be a multiple of 32 (YOLO's largest stride), got {imgsz}")

    source = m.layout.run_dir(export.id) / "files"
    config = {"epochs": epochs, "imgsz": imgsz, "batch": batch, "seed": seed, "workers": workers, "patience": 0}
    with m.writer(step="yolo.train", config=config, inputs=[export.id], tags=tags,
                  model=model.stem, checkpoint=model.name) as w:
        w.add_provenance(checkpoint_sha256=file_sha256(model))
        data = w.tmp_dir / "data"
        _copy_export(source, data)
        spec = {"verb": "train", "data": str(data / "data.yaml"), "model": str(model), **config}
        spec_file = w.tmp_dir / "spec.json"
        spec_file.write_text(json.dumps(spec))
        run_job(JOB_DIR, spec_file)

        report = json.loads((w.tmp_dir / "job.json").read_text())
        trained = Path(report["save_dir"])
        if trained.resolve() != (w.tmp_dir / "train").resolve():
            raise RuntimeError(f"The job trained into {trained}, not {w.tmp_dir / 'train'}")
        missing = sorted(set(METRICS) - set(report["metrics"]))
        if missing:
            raise RuntimeError(f"The job's metrics lack {missing}; it reported {sorted(report['metrics'])}")
        files = w.tmp_dir / "files"
        files.mkdir()
        kept = {"best": "weights/best.pt", "last": "weights/last.pt", "results": "results.csv", "args": "args.yaml"}
        for name, rel in kept.items():
            if not (trained / rel).is_file():
                raise RuntimeError(f"The job left no {rel} in {trained}")
            shutil.move(trained / rel, files / Path(rel).name)
        w.add_provenance(**report["versions"])

        values = {stored: report["metrics"][theirs] for theirs, stored in METRICS.items()}
        values |= {"seconds": report["seconds"], "seconds_per_epoch": report["seconds"] / epochs,
                   "peak_vram_gib": report["peak_vram_gib"]}
        w.write("metrics", pa.table({
            "name": list(values), "class": pa.nulls(len(values), pa.string()),
            "value": pa.array([float(v) for v in values.values()], pa.float64()),
        }))
        w.write("artifacts", pa.table({
            "name": list(kept), "path": [f"files/{Path(rel).name}" for rel in kept.values()],
        }))
        print(f"yolo.train: {epochs} epochs in {report['seconds'] / 60:.1f} min, "
              f"val mAP50-95 {values['val/mAP50-95']:.3f} (SAM3's val labels), peak VRAM {report['peak_vram_gib']:.1f} GiB")
        for leftover in (data, trained, spec_file, w.tmp_dir / "job.json"):
            if leftover.is_dir():
                shutil.rmtree(leftover)
            else:
                leftover.unlink()
    return m.run(w.run_id)


def _copy_export(source: Path, target: Path) -> None:
    """The export's files under `target`: images hard-linked (no extra disk), labels and
    data.yaml copied (Ultralytics writes its labels cache beside them)."""
    for path in sorted(source.rglob("*")):
        dest = target / path.relative_to(source)
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif path.suffix == ".jpg":
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)


def predict(
    m: Master,
    model: Run | str | Path,
    frames: Run,
    *,
    conf: float = 0.001,
    max_det: int = 100,
    imgsz: int,
    batch: int = 16,
    classes: list[str] | None = None,
    tags: dict[str, Any] | None = None,
) -> Run:
    """The detector's boxes on every frame of a frame set (GPU), as an instances run.

    model:   a yolo.train run (its best.pt), or a checkpoint file (e.g. stock yolo26s.pt).
    conf:    boxes scoring below it are not kept. COCO scoring wants the low tail: 0.001.
    max_det: at most this many boxes per frame (COCO's evaluation counts 100).
    imgsz:   inference size; use the size the model was trained at.
    classes: keep only boxes of these class names (e.g. ["person"] from a stock COCO model);
             all classes when None.
    """
    if isinstance(model, Run):
        if model.step != "yolo.train":
            raise TypeError(f"model must be a yolo.train run or a checkpoint file; {model.id} is {model.step}")
        weights, inputs = m.layout.run_dir(model.id) / "files" / "best.pt", [model.id, frames.id]
    else:
        weights, inputs = Path(model).resolve(), [frames.id]
    if not weights.is_file():
        raise FileNotFoundError(f"No YOLO weights at {weights}")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf < 1:
        raise ValueError(f"conf must be in [0, 1), got {conf!r}")
    for name, value, low in (("max_det", max_det, 1), ("imgsz", imgsz, 32), ("batch", batch, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < low:
            raise ValueError(f"{name} must be an integer of at least {low}, got {value!r}")
    if classes is not None and (not classes or any(not isinstance(c, str) or not c for c in classes)):
        raise ValueError(f"classes must be a non-empty list of class names or None, got {classes!r}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"predict needs a frame set; {frames.id} wrote {frames.tables}")

    config = {"conf": conf, "max_det": max_det, "imgsz": imgsz, "batch": batch, "classes": classes}
    with m.writer(step="yolo.predict", config=config, inputs=inputs, tags=tags,
                  model=weights.stem if not isinstance(model, Run) else model.id, checkpoint=weights.name) as w:
        w.add_provenance(checkpoint_sha256=file_sha256(weights))
        images = cached_frames(m, frames)
        spec = {"verb": "predict", "model": str(weights), "images": [str(p) for _, _, p in images], **config}
        spec_file = w.tmp_dir / "spec.json"
        spec_file.write_text(json.dumps(spec))
        run_job(JOB_DIR, spec_file)

        report = json.loads((w.tmp_dir / "job.json").read_text())
        found = [d for d in report["detections"] if classes is None or d["class"] in classes]
        n = len(found)
        w.add_provenance(**report["versions"])
        w.write("instances", pa.table({
            "instance_idx": pa.array(range(n), pa.int64()),
            "media_id": [images[d["image"]][0] for d in found],
            "frame_idx": pa.array([images[d["image"]][1] for d in found], pa.int64()),
            "class": [d["class"] for d in found],
            "score": pa.array([float(d["score"]) for d in found], pa.float64()),
            **{c: pa.array([float(d["box"][k]) for d in found], pa.float64()) for k, c in enumerate(("x0", "y0", "x1", "y1"))},
            "mask_rle": pa.nulls(n, pa.string()),
            "keypoints": pa.nulls(n),
            "track_id": pa.nulls(n, pa.int64()),
        }))
        print(f"yolo.predict: {n} boxes on {len(images)} frames in {report['seconds']:.0f} s")
        spec_file.unlink()  # absolute paths to cached frames: derived, not provenance
        (w.tmp_dir / "job.json").unlink()
    return m.run(w.run_id)
