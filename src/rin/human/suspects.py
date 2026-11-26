"""human.suspects: boxes worth a second look before a person signs a frame off (D-33).

Starting a check from a model's boxes makes it fast, and makes wrong boxes easy to pass: the
first check of the test set kept 99.4% of SAM3's boxes, bags on empty bleachers included
(M3 step 5). These rules point a person at the boxes most likely to be wrong. A flag is a
question, never a verdict: nothing is removed here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from pycocotools import mask as mask_util

from rin.master import Master
from rin.run import Run


def suspects(
    m: Master,
    frames: Run,
    boxes: Run,
    *,
    masks: Run | None = None,
    above: float | None = None,
    same_spot_iou: float = 0.5,
    same_spot_share: float = 0.5,
    same_spot_min_frames: int = 5,
    duplicate_iou: float = 0.7,
    mask_gap_px: int = 20,
) -> list[dict[str, Any]]:
    """Flag boxes of run `boxes` on the frames of frame set `frames`. Each flag: {media_id, frame_idx, x0, y0, x1, y1
    (the flagged box, in [0, 1]), reason}. One box can be flagged for several reasons.
    above: only boxes scoring above it (and boxes without a score), e.g. the gate a check
    starts from (D-33).

    same spot:  on a fixed camera, a box that overlaps (IoU >= same_spot_iou) a box on at least
                `same_spot_share` of that camera's frames of the same day, and on at least
                `same_spot_min_frames` frames. Bags, coats and chairs stay put; people mostly
                don't (a seated crowd does, so expect people among these).
    duplicate:  two boxes on one frame with IoU >= duplicate_iou.
    split mask: needs `masks`, a run with masks (SAM3). A box matching one of its detections
                (IoU >= 0.9 with the box around that mask) whose mask has an empty gap of at
                least `mask_gap_px` pixels across it: the box spans two pieces, often two
                people and the floor between them. When `masks` stored SAM3's detector box
                (box="detector"), the flag carries it as `suggested` (x0, y0, x1, y1): on the
                test set it sat on one person in 16 of 16 split masks looked at (M3 step 5).
    """
    if "frame_sets" not in frames.tables:
        raise TypeError(f"suspects needs a frame set; {frames.id} wrote {frames.tables}")
    if "instances" not in boxes.tables:
        raise TypeError(f"suspects needs a run with boxes; {boxes.id} wrote {boxes.tables}")
    if masks is not None and "instances" not in masks.tables:
        raise TypeError(f"masks must be a run with boxes and masks; {masks.id} wrote {masks.tables}")
    rows = m.sql(
        """SELECT i.media_id, i.frame_idx, i.x0, i.y0, i.x1, i.y1, d.site, d.camera,
                  CAST(d.started_at AS DATE) AS day
           FROM instances i JOIN media d USING (media_id)
           JOIN frame_sets f ON f.media_id = i.media_id AND f.frame_idx = i.frame_idx AND f.run_id = ?
           WHERE i.run_id = ? AND (i.score IS NULL OR ? IS NULL OR i.score > ?)
           ORDER BY i.media_id, i.frame_idx, i.instance_idx""",
        [frames.id, boxes.id, above, above],
    ).fetchall()
    frames_per_camera_day: dict[tuple, int] = defaultdict(int)
    for site, camera, day, n in m.sql(
        """SELECT d.site, d.camera, CAST(d.started_at AS DATE), count(*)
           FROM frame_sets f JOIN media d USING (media_id) WHERE f.run_id = ? GROUP BY ALL""",
        [frames.id],
    ).fetchall():
        frames_per_camera_day[(site, camera, day)] = n

    flags: list[dict[str, Any]] = []

    def flag(row: tuple, reason: str) -> None:
        flags.append(dict(zip(("media_id", "frame_idx", "x0", "y0", "x1", "y1"), row[:6], strict=True)) | {"reason": reason})

    # Same spot, per camera and day. Recordings without a camera or start time are skipped:
    # nothing says which frames share a view.
    by_camera_day: dict[tuple, list[tuple]] = defaultdict(list)
    for row in rows:
        if row[7] is not None and row[8] is not None:
            by_camera_day[(row[6], row[7], row[8])].append(row)
    for key, group in by_camera_day.items():
        b = np.array([r[2:6] for r in group])
        frame_of = [(r[0], r[1]) for r in group]
        overlaps = _iou(b, b) >= same_spot_iou
        need = max(same_spot_min_frames, same_spot_share * frames_per_camera_day[key])
        for i, row in enumerate(group):
            n = len({frame_of[j] for j in np.flatnonzero(overlaps[i])})
            if n >= need:
                flag(row, f"same spot on {n} of {frames_per_camera_day[key]} frames")

    # Duplicates, per frame.
    by_frame: dict[tuple, list[tuple]] = defaultdict(list)
    for row in rows:
        by_frame[(row[0], row[1])].append(row)
    for group in by_frame.values():
        if len(group) < 2:
            continue
        overlaps = _iou(np.array([r[2:6] for r in group]), np.array([r[2:6] for r in group]))
        np.fill_diagonal(overlaps, 0)
        for i, row in enumerate(group):
            if overlaps[i].max() >= duplicate_iou:
                flag(row, f"duplicate (IoU {overlaps[i].max():.2f})")

    if masks is not None:
        detector = masks.config.get("box") == "detector"
        for row, gap, stored in _split_masks(m, masks, by_frame, mask_gap_px):
            flag(row, f"split mask ({gap} px gap)")
            if detector:
                flags[-1]["suggested"] = stored

    counts: dict[str, int] = defaultdict(int)
    for f in flags:
        counts[f["reason"].split(" (")[0].split(" on ")[0]] += 1
    print(f"human.suspects: {len(flags)} flags on {len({(f['media_id'], f['frame_idx']) for f in flags})} frames: {dict(counts)}")
    return flags


def _split_masks(
    m: Master, masks: Run, by_frame: dict[tuple, list[tuple]], gap_px: int
) -> list[tuple[tuple, int, list[float]]]:
    """Boxes of `by_frame` that match a detection of `masks` whose mask has an empty gap of at
    least `gap_px` pixels between two of its pieces, across or down: (row, gap, the box
    `masks` stored for that detection)."""
    sizes = dict((mid, (w, h)) for mid, w, h in m.sql("SELECT media_id, width, height FROM media").fetchall())
    detections: dict[tuple, list[tuple[str, list[float]]]] = defaultdict(list)
    for media_id, frame_idx, rle, *box in m.sql(
        "SELECT media_id, frame_idx, mask_rle, x0, y0, x1, y1 FROM instances WHERE run_id = ? AND mask_rle IS NOT NULL",
        [masks.id],
    ).fetchall():
        if (media_id, frame_idx) in by_frame:
            detections[(media_id, frame_idx)].append((rle, box))
    if not detections:
        raise ValueError(f"Run {masks.id} has no masks on the frames of the flagged run")
    found = []
    for key, group in by_frame.items():
        if key not in detections:
            continue
        width, height = sizes[key[0]]
        decoded = [mask_util.decode({"size": [height, width], "counts": rle.encode()}) for rle, _ in detections[key]]
        mask_boxes = np.array([_mask_box(d, width, height) for d in decoded])
        overlaps = _iou(np.array([r[2:6] for r in group]), mask_boxes)
        for i, row in enumerate(group):
            j = int(overlaps[i].argmax())
            if overlaps[i, j] < 0.9:
                continue
            gap = max(_largest_gap(decoded[j].any(axis=0)), _largest_gap(decoded[j].any(axis=1)))
            if gap >= gap_px:
                found.append((row, gap, detections[key][j][1]))
    return found


def _mask_box(mask: np.ndarray, width: int, height: int) -> list[float]:
    """The tight box around a mask's pixels, in [0, 1] (pixel edges)."""
    cols, rows = np.flatnonzero(mask.any(axis=0)), np.flatnonzero(mask.any(axis=1))
    if not len(cols):
        return [0.0, 0.0, 0.0, 0.0]
    return [cols[0] / width, rows[0] / height, (cols[-1] + 1) / width, (rows[-1] + 1) / height]


def _largest_gap(occupied: np.ndarray) -> int:
    """The longest run of empty pixels between two occupied ones."""
    idx = np.flatnonzero(occupied)
    return int(np.diff(idx).max() - 1) if len(idx) > 1 else 0


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a against every box in b (x0, y0, x1, y1)."""
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
