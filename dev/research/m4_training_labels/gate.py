"""SAM3's precision and recall against the checked test set, per score gate (M4 step 2).

    uv run python dev/research/m4_training_labels/gate.py

A SAM3 box is right when it matches a checked person box at IoU >= 0.5 (greedy, highest score
first, each checked box matched once). An unmatched SAM3 box at least half inside a `crowd`
box is ignored, as scoring will ignore it (D-34). Sizes are box heights in 1080p pixels.
"""

from collections import defaultdict

import numpy as np

from rin import Master

SAM3, CHECKED = "20261126-205940-sam3.concept-11ee", "20261127-094340-human.annotate-64f9"
GATES = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box a (4,) with boxes b (n, 4)."""
    w = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = w * h
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter)


def inside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of box a's area inside each of boxes b."""
    w = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]))


m = Master("workspace/store")
frames = m.sql("SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ?", [CHECKED]).fetchall()
boxes = defaultdict(lambda: {"person": [], "crowd": [], "sam3": []})
for run, kind in ((CHECKED, None), (SAM3, "sam3")):
    for media_id, idx, cls, score, *xyxy in m.sql(
        "SELECT media_id, frame_idx, class, score, x0, y0, x1, y1 FROM instances WHERE run_id = ?", [run]
    ).fetchall():
        boxes[(media_id, idx)][kind or cls].append(([score] if kind else []) + xyxy)

n_truth = sum(len(boxes[f]["person"]) for f in frames)
print(f"{len(frames)} checked frames, {n_truth} people, {sum(len(boxes[f]['crowd']) for f in frames)} crowd boxes\n")
print("gate   boxes   right  wrong  ignored  precision  recall   F1    wrong<40px  missed<40px")
for gate in GATES:
    tp = fp = ign = fp_small = 0
    missed_small = 0
    for f in frames:
        truth = np.array(boxes[f]["person"]).reshape(-1, 4)
        crowd = np.array(boxes[f]["crowd"]).reshape(-1, 4)
        dets = sorted((d for d in boxes[f]["sam3"] if d[0] >= gate), key=lambda d: -d[0])
        used = np.zeros(len(truth), bool)
        for d in dets:
            box = np.array(d[1:])
            if len(truth):
                o = iou(box, truth)
                o[used] = 0
                j = int(o.argmax())
                if o[j] >= 0.5:
                    used[j] = True
                    tp += 1
                    continue
            if len(crowd) and inside(box, crowd).max() >= 0.5:
                ign += 1
                continue
            fp += 1
            fp_small += (box[3] - box[1]) * 1080 < 40
        missed_small += int(((truth[~used, 3] - truth[~used, 1]) * 1080 < 40).sum())
    p, r = tp / max(tp + fp, 1), tp / n_truth
    print(f"{gate:4.2f}  {tp + fp + ign:6d}  {tp:6d} {fp:6d} {ign:7d}     {p:.3f}    {r:.3f}  {2 * p * r / (p + r):.3f}"
          f"  {fp_small:9d}  {missed_small:10d}")
