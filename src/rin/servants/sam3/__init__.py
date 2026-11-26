"""sam3.concept: every instance of each class on each frame, found by SAM3 from a text prompt.

The caller (this file) runs in the core environment. The job (job.py) runs in this folder's own
environment, on the user's engine in vendors/sam3_engine, and imports nothing from Rin:

    caller: open the run, cache the frames, write spec.json  ->  job: write job.json
    caller: turn pixels into [0, 1] boxes, write the instances table, file the run
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa

from rin.master import Master
from rin.run import Run
from rin.servants.frame_cache import cached_frames
from rin.servants.launch import run_job
from rin.store import vendor_commit

JOB_DIR = Path(__file__).resolve().parent
ENGINE_DIR = JOB_DIR.parents[3] / "vendors" / "sam3_engine"


def concept(
    m: Master,
    frames: Run,
    *,
    prompts: dict[str, str],
    checkpoint: str | Path,
    score_floor: float = 0.1,
    masks: bool = True,
    box: str = "detector",
    images_per_session: int = 8,
) -> Run:
    """Detect every instance of each class on every frame of a frame set (GPU). Each frame is
    labelled on its own (no tracking).

    prompts:            class -> the text SAM3 is asked for, e.g. {"person": "person"}. One text
                        per class: two texts would find the same object twice.
    checkpoint:         SAM3's image checkpoint file (sam3.pt), inside the workspace.
    score_floor:        detections scoring at or below it are not stored. Keep it low: the
                        real score gate is chosen later, without running SAM3 again (D-26).
    masks:              store each detection's mask as COCO RLE at the media's full size.
    box:                "detector" (D-36): SAM3's own box, predicted separately from the mask.
                        "mask": the tight box around the mask; when a mask has two pieces (a
                        stray patch on another person) it spans both.
    images_per_session: images whose features sit on the GPU at once. 8 fits in 16 GB (v1).
    """
    if not isinstance(prompts, dict) or not prompts or not all(
        isinstance(c, str) and c.strip() and isinstance(t, str) and t.strip() for c, t in prompts.items()
    ):
        raise ValueError(f"prompts must map class names to one prompt text each, got {prompts!r}")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No SAM3 checkpoint at {checkpoint}")
    if not isinstance(score_floor, (int, float)) or isinstance(score_floor, bool) or not 0 <= score_floor < 1:
        raise ValueError(f"score_floor must be in [0, 1), got {score_floor!r}")
    if not isinstance(masks, bool):
        raise ValueError(f"masks must be True or False, got {masks!r}")
    if box not in ("mask", "detector"):
        raise ValueError(f"box must be 'mask' or 'detector', got {box!r}")
    if not isinstance(images_per_session, int) or isinstance(images_per_session, bool) or images_per_session < 1:
        raise ValueError(f"images_per_session must be a positive integer, got {images_per_session!r}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"sam3.concept needs a frame set; {frames.id} wrote {frames.tables}")
    if not m.sql("SELECT 1 FROM frame_sets WHERE run_id = ? LIMIT 1", [frames.id]).fetchall():
        raise ValueError(f"Frame set {frames.id} is empty")

    config = {
        "prompts": prompts, "score_floor": score_floor, "masks": masks, "box": box,
        "images_per_session": images_per_session,
    }
    with m.writer(
        step="sam3.concept", config=config, inputs=[frames.id],
        model="sam3", checkpoint=m.layout.relative_media_path(checkpoint),
    ) as w:
        w.add_provenance(checkpoint_sha256=_sha256(checkpoint), sam3_engine=vendor_commit(ENGINE_DIR))
        images = cached_frames(m, frames)
        sizes = dict(
            (media_id, (width, height))
            for media_id, width, height in m.sql("SELECT media_id, width, height FROM media").fetchall()
        )
        spec = {
            "checkpoint": str(checkpoint), **config,
            "images": [
                {"path": str(path), "width": sizes[media_id][0], "height": sizes[media_id][1]}
                for media_id, _, path in images
            ],
        }
        spec_file = w.tmp_dir / "spec.json"
        spec_file.write_text(json.dumps(spec))
        run_job(JOB_DIR, spec_file)

        report = json.loads((w.tmp_dir / "job.json").read_text())
        found = report["detections"]
        rows = {name: [] for name in ("media_id", "frame_idx", "class", "score", "x0", "y0", "x1", "y1", "mask_rle")}
        for d in found:
            media_id, frame_idx, _ = images[d["image"]]
            width, height = sizes[media_id]
            x0, y0, x1, y1 = d["box"]
            for name, value in (
                ("media_id", media_id), ("frame_idx", frame_idx), ("class", d["class"]), ("score", d["score"]),
                ("x0", x0 / width), ("y0", y0 / height), ("x1", x1 / width), ("y1", y1 / height),
                ("mask_rle", d["mask_rle"]),
            ):
                rows[name].append(value)
        if masks and any(not r for r in rows["mask_rle"]):
            raise RuntimeError("The job wrote a detection without a mask, although masks=True")
        n = len(found)
        w.add_provenance(**report["versions"])
        w.write("instances", pa.table({
            "instance_idx": pa.array(range(n), pa.int64()),
            **{name: rows[name] for name in ("media_id", "frame_idx", "class")},
            "score": pa.array(rows["score"], pa.float64()),
            **{name: pa.array(rows[name], pa.float64()) for name in ("x0", "y0", "x1", "y1")},
            "mask_rle": pa.array(rows["mask_rle"], pa.string()),
            "keypoints": pa.nulls(n),
            "track_id": pa.nulls(n, pa.int64()),
        }))
        print(
            f"sam3.concept: {n} detections on {len(images)} frames; job {report['seconds']:.0f} s "
            f"({report['seconds'] / len(images):.2f} s/frame), peak VRAM {report['peak_vram_gib']:.1f} GiB"
        )
        spec_file.unlink()  # absolute paths to cached frames: derived, not provenance
        (w.tmp_dir / "job.json").unlink()  # its content is now the instances table
    return m.run(w.run_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()
