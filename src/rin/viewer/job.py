"""FiftyOne side of the viewer (D-16). Runs in this folder's own environment; imports nothing from Rin.

    uv run --project src/rin/viewer python job.py <spec.json>

spec["command"] is one of:
- "load": build a throwaway FiftyOne dataset `name` from frames (cached JPEGs), each frame's
  boxes from runs as read-only fields, and an empty field a person draws boxes into.
- "dump": write the boxes drawn in `field` of dataset `name`, and each frame's tags, to `out`.

The App itself is started by hand while the person draws:
    uv run --project src/rin/viewer fiftyone app launch <name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fiftyone as fo


def load(spec: dict) -> None:
    name = spec["name"]
    if fo.dataset_exists(name):
        if not spec["replace"]:
            raise SystemExit(f"A FiftyOne dataset {name!r} exists; pass replace=True to rebuild it")
        fo.delete_dataset(name)
    dataset = fo.Dataset(name, persistent=True)  # persistent: survives restarting the App
    samples = []
    for frame in spec["frames"]:
        sample = fo.Sample(filepath=frame["path"], rin_media_id=frame["media_id"], rin_frame_idx=frame["frame_idx"])
        sample.metadata = fo.ImageMetadata(width=frame["width"], height=frame["height"])
        for field, boxes in frame["labels"].items():
            sample[field] = fo.Detections(detections=[
                fo.Detection(label=b["class"], bounding_box=b["xywh"], confidence=b["score"]) for b in boxes
            ])
        if spec["draw_field"]:
            sample[spec["draw_field"]] = fo.Detections(detections=[])
        samples.append(sample)
    dataset.add_samples(samples)

    if spec["draw_field"]:
        field = spec["draw_field"]
        schema = dataset.generate_label_schemas(field, scan_samples=False)
        # The class picker: a radio list (FiftyOne requires a radio or dropdown for `classes`).
        schema.update(classes=spec["classes"], component="radio", default=spec["classes"][0])
        dataset.set_label_schemas({field: schema})
        dataset.activate_label_schemas(field)
    print(f"FiftyOne dataset {name!r}: {len(samples)} frames, label fields {sorted(spec['label_fields'])}, "
          f"draw into {spec['draw_field']!r}", flush=True)


def dump(spec: dict) -> None:
    if not fo.dataset_exists(spec["name"]):
        raise SystemExit(f"No FiftyOne dataset {spec['name']!r}")
    dataset = fo.load_dataset(spec["name"])
    field = spec["field"]
    if field not in dataset.get_field_schema():
        raise SystemExit(f"Dataset {spec['name']!r} has no field {field!r}")
    frames = []
    for sample in dataset.select_fields(["rin_media_id", "rin_frame_idx", field]):
        drawn = sample[field].detections if sample[field] is not None else []
        frames.append({
            "media_id": sample["rin_media_id"], "frame_idx": sample["rin_frame_idx"], "tags": list(sample.tags),
            "boxes": [{"class": d.label, "xywh": list(d.bounding_box)} for d in drawn],
        })
    Path(spec["out"]).write_text(json.dumps({"frames": frames, "fiftyone": fo.__version__}))
    print(f"{sum(len(f['boxes']) for f in frames)} boxes on {len(frames)} frames read from {spec['name']!r}.{field}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: job.py <spec.json>")
    spec = json.loads(Path(sys.argv[1]).read_text())
    {"load": load, "dump": dump}[spec["command"]](spec)
