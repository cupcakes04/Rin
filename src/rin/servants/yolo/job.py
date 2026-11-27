"""YOLO training job. Runs in this folder's own environment; imports nothing from Rin.

    uv run --project src/rin/servants/yolo python job.py <spec.json>

spec.json says which verb:
- "train" (data, model, epochs, imgsz, batch, seed, workers): trains into <spec folder>/train/
  (Ultralytics' own layout: weights/best.pt, weights/last.pt, results.csv, args.yaml). job.json:
  best.pt's val metrics, time, peak VRAM, versions.
- "predict" (model, images, conf, max_det, imgsz, batch): job.json holds every detection as
  {image (index into images), class, score, box (x0, y0, x1, y1 in [0, 1])}, time, versions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def main(spec_file: Path) -> None:
    spec = json.loads(spec_file.read_text())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to the YOLO job")
    {"train": train, "predict": predict}[spec["verb"]](spec, spec_file.parent)


def predict(spec: dict, out_dir: Path) -> None:
    model = YOLO(spec["model"])
    paths = spec["images"]
    detections = []
    started = time.perf_counter()
    for s in range(0, len(paths), spec["batch"]):
        results = model.predict(
            paths[s : s + spec["batch"]], conf=spec["conf"], max_det=spec["max_det"], imgsz=spec["imgsz"],
            device=0, verbose=False,
        )
        for k, r in enumerate(results):
            for box, score, cls in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), strict=True):
                detections.append({"image": s + k, "class": r.names[int(cls)], "score": score, "box": box})
    (out_dir / "job.json").write_text(json.dumps({
        "detections": detections, "seconds": time.perf_counter() - started,
        "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
    }))


def train(spec: dict, out_dir: Path) -> None:
    # Ultralytics' mixed-precision check loads yolo26n.pt from the working folder and downloads
    # it when missing; the checkpoint folder already holds it, so nothing lands in the repo.
    os.chdir(Path(spec["model"]).parent)

    model = YOLO(spec["model"])
    started = time.perf_counter()
    model.train(
        data=spec["data"], epochs=spec["epochs"], imgsz=spec["imgsz"], batch=spec["batch"],
        seed=spec["seed"], deterministic=True, workers=spec["workers"],
        patience=0,          # no early stop: every run trains all its epochs; best.pt is picked by val
        plots=False,         # no plots (they also download a font)
        project=str(out_dir), name="train", exist_ok=True, device=0,
    )
    seconds = time.perf_counter() - started
    trainer = model.trainer
    # After the last epoch Ultralytics validates best.pt again; trainer.metrics holds that pass.
    metrics = {k: float(v) for k, v in trainer.metrics.items()}
    (out_dir / "job.json").write_text(json.dumps({
        "save_dir": str(trainer.save_dir),
        "metrics": metrics,
        "best_fitness": float(trainer.best_fitness) if trainer.best_fitness is not None else None,
        "seconds": seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
    }))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: job.py <spec.json>")
    main(Path(sys.argv[1]))
