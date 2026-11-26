"""SAM3 concept-detection job. Runs in this folder's own environment; imports nothing from Rin.

    uv run --project src/rin/servants/sam3 python job.py <spec.json>

Reads spec.json (checkpoint, prompts, score_floor, masks, images_per_session, images) and
writes job.json next to it: every detection as {image, class, score, box, mask_rle}, plus
library versions and the job's own cost. `box` is in pixels (x0, y0, x1, y1, pixel edges),
tight around the mask; `mask_rle` is COCO counts, or null without masks.
"""

from __future__ import annotations

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import sam3_engine as se
import torch
from PIL import Image
from pycocotools import mask as mask_util


def main(spec_file: Path) -> None:
    spec = json.loads(spec_file.read_text())
    images, step = spec["images"], spec["images_per_session"]
    engine = se.Sam3Engine(se.EngineConfig(
        image_checkpoint_path=spec["checkpoint"],
        detection_score_threshold=spec["score_floor"],
    ))
    found: list[dict] = []
    started = time.perf_counter()
    with engine:
        # One session per few images: opening a session runs the backbone on all its images at
        # once and keeps their features on the GPU (v1 ran out of 16 GB without chunking).
        for s in range(0, len(images), step):
            chunk = images[s : s + step]
            pictures = [_load(item) for item in chunk]
            with engine.open_image_session(images=pictures) as session:
                for i in range(len(chunk)):
                    for cls, text in spec["prompts"].items():
                        # A detection whose mask is empty at full size is dropped; the engine
                        # logs how many (4 of 34,000 on the test day, all scoring 0.5 or less).
                        for obj in session.detect(
                            image_idx=i, text=text, exemplar_boxes_xyxy_px=None,
                            exemplar_labels=None, replace_existing_objects=False,
                            keep_empty_mask_detections=False,
                        ):
                            found.append({
                                "image": s + i, "class": cls, "score": obj.score,
                                "box": list(obj.mask_box_xyxy_px),  # tight around the mask
                                "mask_rle": _rle(obj.mask) if spec["masks"] else None,
                            })
            done = s + len(chunk)
            if (s // step) % 10 == 9 or done == len(images):
                rate = done / (time.perf_counter() - started)
                print(f"{done}/{len(images)} images, {rate:.2f} images/s, {len(found)} detections", flush=True)

    (spec_file.parent / "job.json").write_text(json.dumps({
        "detections": found,
        "versions": {"torch": torch.__version__, "pycocotools": version("pycocotools")},
        "seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }))


def _load(item: dict) -> Image.Image:
    with Image.open(item["path"]) as image:
        picture = image.convert("RGB")
    if picture.size != (item["width"], item["height"]):
        raise RuntimeError(f"{item['path']} is {picture.size}, but its media is {item['width']}x{item['height']}")
    return picture


def _rle(mask: np.ndarray) -> str:
    """COCO compressed RLE counts. pycocotools wants uint8 in column-major (Fortran) order."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: job.py <spec.json>")
    main(Path(sys.argv[1]))
