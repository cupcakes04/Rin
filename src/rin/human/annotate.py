"""human.annotate: boxes a person drew (in the viewer) -> a run with that person in `who`."""

from __future__ import annotations

import pyarrow as pa

from rin import viewer
from rin.master import Master
from rin.run import Run

def annotate(m: Master, frames: Run, *, view: str, field: str, who: str, done_tag: str = "done") -> Run:
    """File the frames of viewer dataset `view` that the person tagged `done_tag`, with the
    boxes drawn on them in `field`.

    The run writes two tables:
    - frame_sets: the finished frames. "Finished" means every object on it has a box, so a
      finished frame with no box is a frame with nothing on it, which a missing frame is not.
    - instances: the boxes on those frames; no score (a person's box).
    frames: the frame set the view was built from; a finished frame outside it is refused.
    Boxes on frames not tagged finished are refused too: tag the frame, or delete the boxes.
    A box drawn past the image edge is clipped to it (the count is printed).
    who:    the person who drew them (D-27).
    The view is a copy: once its frames are filed, it can be deleted.
    """
    if not isinstance(who, str) or not who.strip():
        raise ValueError(f"who must name the person who drew the boxes, got {who!r}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"annotate needs the frame set the boxes were drawn on; {frames.id} wrote {frames.tables}")

    config = {"view": view, "field": field, "done_tag": done_tag}
    with m.writer(step="human.annotate", config=config, inputs=[frames.id], who=who) as w:
        drawn = viewer.drawn_boxes(view, field, w.tmp_dir)
        allowed = set(map(tuple, m.sql(
            "SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ?", [frames.id]
        ).fetchall()))
        done = sorted(
            (f for f in drawn["frames"] if done_tag in f["tags"]), key=lambda f: (f["media_id"], f["frame_idx"])
        )
        if not done:
            raise ValueError(f"No frame of view {view!r} is tagged {done_tag!r}; nothing to file")
        outside = [(f["media_id"], f["frame_idx"]) for f in done if (f["media_id"], f["frame_idx"]) not in allowed]
        if outside:
            raise ValueError(f"{len(outside)} finished frames are not in {frames.id}, e.g. {outside[:3]}")
        unfinished = [(f["media_id"], f["frame_idx"]) for f in drawn["frames"] if f["boxes"] and done_tag not in f["tags"]]
        if unfinished:
            raise ValueError(
                f"{len(unfinished)} frames have boxes but no {done_tag!r} tag, e.g. {unfinished[:3]}: "
                "tag them when finished, or delete their boxes"
            )

        rows: dict[str, list] = {k: [] for k in ("media_id", "frame_idx", "class", "x0", "y0", "x1", "y1")}
        clipped = 0
        for frame in done:
            for box in frame["boxes"]:
                if not box["class"]:
                    raise ValueError(f"A box on {frame['media_id']} frame {frame['frame_idx']} has no class")
                x, y, bw, bh = box["xywh"]
                raw = (x, y, x + bw, y + bh)
                # A drag may run past the image edge (FiftyOne allows it, e.g. for a person cut
                # off by the frame); what is visible ends at the edge, so clip. A box with no
                # area left inside the image is still refused by the store's box check.
                corners = [min(max(v, 0.0), 1.0) for v in raw]
                clipped += corners != list(raw)
                for name, value in zip(
                    ("media_id", "frame_idx", "class", "x0", "y0", "x1", "y1"),
                    (frame["media_id"], frame["frame_idx"], box["class"], *corners), strict=True,
                ):
                    rows[name].append(value)
        n = len(rows["media_id"])
        w.add_provenance(fiftyone=drawn["fiftyone"])
        w.write("frame_sets", pa.table({
            "media_id": [f["media_id"] for f in done],
            "frame_idx": pa.array([f["frame_idx"] for f in done], pa.int64()),
            "rank": pa.nulls(len(done), pa.int64()),
        }))
        w.write("instances", pa.table({
            "instance_idx": pa.array(range(n), pa.int64()),
            "media_id": rows["media_id"], "frame_idx": pa.array(rows["frame_idx"], pa.int64()),
            "class": rows["class"], "score": pa.nulls(n, pa.float64()),
            **{c: pa.array(rows[c], pa.float64()) for c in ("x0", "y0", "x1", "y1")},
            "mask_rle": pa.nulls(n, pa.string()), "keypoints": pa.nulls(n), "track_id": pa.nulls(n, pa.int64()),
        }))
        print(
            f"human.annotate: {n} boxes on {len(done)} finished frames (of {len(drawn['frames'])} in the view), "
            f"by {who}; {clipped} boxes ran past the image edge and were clipped to it"
        )
    return m.run(w.run_id)

