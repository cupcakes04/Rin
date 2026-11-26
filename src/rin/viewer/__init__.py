"""viewer: frames and boxes shown in FiftyOne, a throwaway copy (D-16), where a person also draws boxes.

The caller (this file) runs in the core environment; job.py runs in this folder's own
environment (FiftyOne and its database) and imports nothing from Rin. Nothing here writes a
run: a view is a copy for looking, never data. What a person draws comes back into the store
through `human.annotate`.

    viewer.show(m, frames, name="check", draw="checked", start_from=sam, start_above=0.5,
                labels=[sam], label_scores=(0.3, 0.5))
    #   then, in a terminal:  uv run --project src/rin/viewer fiftyone app launch check
    #   draw, and tag each frame "done" once every object on it has a box
    human.annotate(m, frames, view="check", field="checked", who="nic")
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
    label_scores: tuple[float, float] | None = None,
    draw: str | None = None,
    start_from: Run | None = None,
    start_above: float = 0.5,
    classes: list[str] | tuple[str, ...] = ("person",),
    replace: bool = False,
) -> None:
    """Build FiftyOne dataset `name` from a frame set, ordered by `rank` when it has one.

    labels:       runs whose boxes on these frames are shown, read-only, one field per run
                  (named `run_<step>_<4 chars>`).
    label_scores: (low, high): of those, only boxes scoring above low and up to high, e.g.
                  (0.3, 0.5) for a hint layer of the boxes a gate leaves out, named
                  `hints_<servant>_<low>_to_<high>`. Boxes without a score are always shown.
    draw:         the name of a field to draw boxes into, with `classes` to pick from;
                  None to only look.
    start_from:   a run whose boxes scoring above `start_above` (and any without a score) are
                  copied into `draw` to start from (D-33): the person deletes, fixes, adds.
    replace:      rebuild `name` if it exists. Refused by default: it may hold unsaved work.

    The dataset remembers the frame set and what it started from; `human.annotate` records them.
    """
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError(f"name must be letters, digits, '-' or '_', got {name!r}")
    if draw is not None and (not isinstance(draw, str) or not NAME.match(draw) or draw.startswith(("run_", "hints_"))):
        raise ValueError(f"draw must be a field name (letters, digits, '-', '_', not 'run_...' or 'hints_...'), got {draw!r}")
    classes = list(classes)
    if draw is not None and (not classes or not all(isinstance(c, str) and c for c in classes)):
        raise ValueError(f"classes must be non-empty class names, got {classes!r}")
    if label_scores is not None and not (
        len(label_scores) == 2 and all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in label_scores)
        and 0 <= label_scores[0] < label_scores[1] <= 1
    ):
        raise ValueError(f"label_scores must be (low, high) with 0 <= low < high <= 1, got {label_scores!r}")
    if start_from is not None and draw is None:
        raise ValueError("start_from needs a draw field to copy the boxes into")
    if not isinstance(start_above, (int, float)) or isinstance(start_above, bool) or not 0 <= start_above < 1:
        raise ValueError(f"start_above must be in [0, 1), got {start_above!r}")
    for run in [*labels, *([start_from] if start_from is not None else [])]:
        if "instances" not in run.tables:
            raise TypeError(f"labels and start_from must be runs with boxes; {run.id} wrote {run.tables}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"show needs a frame set; {frames.id} wrote {frames.tables}")

    order = {
        (media_id, idx): k for k, (media_id, idx) in enumerate(m.sql(
            "SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ? ORDER BY rank NULLS LAST, media_id, frame_idx",
            [frames.id],
        ).fetchall())
    }
    sizes = dict((mid, (w, h)) for mid, w, h in m.sql("SELECT media_id, width, height FROM media").fetchall())
    if label_scores is None:
        fields = {run.id: f"run_{run.step.replace('.', '_')}_{run.id[-4:]}" for run in labels}
    else:  # a hint layer says what it is: whose boxes, which scores
        band = "_to_".join(str(s).replace(".", "_") for s in label_scores)
        fields = {run.id: f"hints_{run.step.split('.')[0]}_{band}" for run in labels}
    if len(set(fields.values())) != len(fields):
        raise ValueError(f"Two label runs would share a field name: {fields}")
    low, high = label_scores if label_scores is not None else (None, None)
    layers = {key: {f: [] for f in fields.values()} for key in order}
    for run in labels:
        for key, box in _boxes(m, run, frames, above=low, up_to=high):
            layers[key][fields[run.id]].append(box)
    start: dict[tuple[str, int], list[dict[str, Any]]] = {key: [] for key in order}
    if start_from is not None:
        for key, box in _boxes(m, start_from, frames, above=start_above, up_to=None):
            start[key].append(box)

    images = sorted(cached_frames(m, frames), key=lambda t: order[(t[0], t[1])])
    spec = {
        "command": "load", "name": name, "replace": replace, "draw_field": draw, "classes": classes,
        "label_fields": sorted(fields.values()),
        "info": {
            "frames": frames.id, "labels": [run.id for run in labels], "label_scores": label_scores,
            "start_from": start_from.id if start_from is not None else None,
            "start_above": start_above if start_from is not None else None,
        },
        "frames": [
            {"path": str(path), "media_id": media_id, "frame_idx": idx,
             "width": sizes[media_id][0], "height": sizes[media_id][1],
             "labels": layers[(media_id, idx)], "start": start[(media_id, idx)]}
            for media_id, idx, path in images
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        spec_file = Path(tmp) / "spec.json"
        spec_file.write_text(json.dumps(spec))
        run_job(JOB_DIR, spec_file)
    print(f"Open it:  uv run --project src/rin/viewer fiftyone app launch {name}")


def _boxes(
    m: Master, run: Run, frames: Run, *, above: float | None, up_to: float | None
) -> list[tuple[tuple[str, int], dict[str, Any]]]:
    """A run's boxes on the frames of a frame set, as ((media_id, frame_idx), box). With a band,
    only boxes scoring above `above` and up to `up_to`; boxes without a score always."""
    rows = m.sql(
        """SELECT i.media_id, i.frame_idx, i.class, i.score, i.x0, i.y0, i.x1, i.y1
           FROM instances i JOIN frame_sets s USING (media_id, frame_idx)
           WHERE i.run_id = ? AND s.run_id = ?
             AND (i.score IS NULL OR ((? IS NULL OR i.score > ?) AND (? IS NULL OR i.score <= ?)))
           ORDER BY i.instance_idx""",
        [run.id, frames.id, above, above, up_to, up_to],
    ).fetchall()
    return [
        ((media_id, idx), {"class": cls, "score": score, "xywh": [x0, y0, x1 - x0, y1 - y0]})
        for media_id, idx, cls, score, x0, y0, x1, y1 in rows
    ]


def drawn_boxes(name: str, field: str, out_dir: Path) -> dict[str, Any]:
    """What a person drew in `field` of FiftyOne dataset `name`: {"frames": [{media_id,
    frame_idx, tags, boxes: [{class, xywh}]}], "info": what `show` built it from,
    "fiftyone": version}. Boxes are [0, 1], x y w h."""
    spec_file = out_dir / "dump_spec.json"
    out = out_dir / "drawn.json"
    spec_file.write_text(json.dumps({"command": "dump", "name": name, "field": field, "out": str(out)}))
    run_job(JOB_DIR, spec_file)
    result = json.loads(out.read_text())
    spec_file.unlink()
    out.unlink()
    return result
