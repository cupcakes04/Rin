"""viewer: frames and boxes shown in FiftyOne, a throwaway copy (D-16), where a person also draws boxes.

The caller (this file) runs in the core environment; job.py runs in this folder's own
environment (FiftyOne and its database) and imports nothing from Rin. Nothing here writes a
run: a view is a copy for looking, never data. What a person draws comes back into the store
through `human.annotate`.

    viewer.show(m, frames, name="pilot", draw="drawn")
    #   then, in a terminal:  uv run --project src/rin/viewer fiftyone app launch pilot
    #   draw, and tag each frame "done" once every object on it has a box
    human.annotate(m, frames, view="pilot", field="drawn", who="nic")
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from rin.master import Master
from rin.run import Run
from rin.servants.frame_cache import cached_frames
from rin.servants.launch import run_job

JOB_DIR = Path(__file__).resolve().parent
NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def show(
    m: Master,
    frames: Run,
    *,
    name: str,
    labels: list[Run] | tuple[Run, ...] = (),
    draw: str | None = None,
    classes: list[str] | tuple[str, ...] = ("person",),
    replace: bool = False,
) -> None:
    """Build FiftyOne dataset `name` from a frame set, ordered by `rank` when it has one.

    labels:  runs whose boxes on these frames are shown, read-only, one field per run
             (named `run_<step>_<4 chars>`).
    draw:    the name of an empty field to draw boxes into, with `classes` to pick from;
             None to only look.
    replace: rebuild `name` if it exists. Refused by default: it may hold unsaved drawing.
    """
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError(f"name must be letters, digits, '-' or '_', got {name!r}")
    if draw is not None and (not isinstance(draw, str) or not NAME.match(draw) or draw.startswith("run_")):
        raise ValueError(f"draw must be a field name (letters, digits, '-', '_', not starting with 'run_'), got {draw!r}")
    classes = list(classes)
    if draw is not None and (not classes or not all(isinstance(c, str) and c for c in classes)):
        raise ValueError(f"classes must be non-empty class names, got {classes!r}")
    for run in labels:
        if "instances" not in run.tables:
            raise TypeError(f"labels must be runs with boxes; {run.id} wrote {run.tables}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"show needs a frame set; {frames.id} wrote {frames.tables}")

    order = {
        (media_id, idx): k for k, (media_id, idx) in enumerate(m.sql(
            "SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ? ORDER BY rank NULLS LAST, media_id, frame_idx",
            [frames.id],
        ).fetchall())
    }
    sizes = dict((mid, (w, h)) for mid, w, h in m.sql("SELECT media_id, width, height FROM media").fetchall())
    fields = {run.id: f"run_{run.step.replace('.', '_')}_{run.id[-4:]}" for run in labels}
    if len(set(fields.values())) != len(fields):
        raise ValueError(f"Two label runs would share a field name: {fields}")
    boxes: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {key: {f: [] for f in fields.values()} for key in order}
    for run in labels:
        for media_id, idx, cls, score, x0, y0, x1, y1 in m.sql(
            """SELECT i.media_id, i.frame_idx, i.class, i.score, i.x0, i.y0, i.x1, i.y1
               FROM instances i JOIN frame_sets s USING (media_id, frame_idx)
               WHERE i.run_id = ? AND s.run_id = ?""",
            [run.id, frames.id],
        ).fetchall():
            boxes[(media_id, idx)][fields[run.id]].append(
                {"class": cls, "score": score, "xywh": [x0, y0, x1 - x0, y1 - y0]}
            )

    images = sorted(cached_frames(m, frames), key=lambda t: order[(t[0], t[1])])
    spec = {
        "command": "load", "name": name, "replace": replace, "draw_field": draw, "classes": classes,
        "label_fields": sorted(fields.values()),
        "frames": [
            {"path": str(path), "media_id": media_id, "frame_idx": idx,
             "width": sizes[media_id][0], "height": sizes[media_id][1], "labels": boxes[(media_id, idx)]}
            for media_id, idx, path in images
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        spec_file = Path(tmp) / "spec.json"
        spec_file.write_text(json.dumps(spec))
        run_job(JOB_DIR, spec_file)
    print(f"Open it:  uv run --project src/rin/viewer fiftyone app launch {name}")


def drawn_boxes(name: str, field: str, out_dir: Path) -> dict[str, Any]:
    """What a person drew in `field` of FiftyOne dataset `name`: {"frames": [{media_id,
    frame_idx, tags, boxes: [{class, xywh}]}], "fiftyone": version}. Boxes are [0, 1], x y w h."""
    spec_file = out_dir / "dump_spec.json"
    out = out_dir / "drawn.json"
    spec_file.write_text(json.dumps({"command": "dump", "name": name, "field": field, "out": str(out)}))
    run_job(JOB_DIR, spec_file)
    result = json.loads(out.read_text())
    spec_file.unlink()
    out.unlink()
    return result
