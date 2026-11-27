"""What each selection method picks from the pool (M4 step 3).

    uv run python dev/research/m4_selection/measure.py

Per method at seed 0 and each budget (first N): share of G299; people per frame and box heights
from D-39's training labels (SAM3 >= 0.5); spread in time (distinct minutes, per camera);
coverage, the mean Euclidean distance (raw DINOv3 CLS) from every candidate to its nearest picked
frame of the same camera, and radius, the largest such distance (what kcenter minimises); and how many frames the methods and seeds share at 1,000.
"""

import numpy as np

from rin import Master

CANDS, EMB, LABELS = (
    "20261127-105927-curate.dedupe-6b65", "20261127-104644-dinov3.embed-32fd", "20261127-111139-sam3.concept-5477"
)
SELECTIONS = {
    "random": ["20261127-132314-curate.select-5da9", "20261127-132314-curate.select-8594", "20261127-132314-curate.select-4375"],
    "kcenter": ["20261127-132314-curate.select-8f97", "20261127-132326-curate.select-5543", "20261127-132338-curate.select-9aab"],
    "kmeans_stratified": ["20261127-132349-curate.select-2d06", "20261127-132359-curate.select-d369", "20261127-132408-curate.select-c75a"],
}
BUDGETS = [100, 250, 500, 1000]

m = Master("workspace/store")
frames = m.sql(
    """SELECT s.media_id, s.frame_idx, d.camera,
              CAST(floor(epoch(d.started_at) / 60 + f.pts / 60) AS BIGINT) AS minute, e.row
       FROM frame_sets s JOIN media d USING (media_id) JOIN frames f USING (media_id, frame_idx)
       JOIN embeddings e ON e.run_id = ? AND e.kind = 'cls' AND e.media_id = s.media_id AND e.frame_idx = s.frame_idx
       WHERE s.run_id = ?""",
    [EMB, CANDS],
).df()
key = list(zip(frames.media_id, frames.frame_idx))
index = {k: i for i, k in enumerate(key)}
boxes = m.sql(
    "SELECT media_id, frame_idx, (y1 - y0) * 1080 AS h FROM instances WHERE run_id = ? AND score >= 0.5", [LABELS]
).df()
people = boxes.groupby(["media_id", "frame_idx"]).size()
frames["people"] = [int(people.get(k, 0)) for k in key]
heights = boxes.groupby(["media_id", "frame_idx"])["h"].apply(list)
vectors = m.run(EMB).vectors().astype(np.float32)
cam_rows = {c: frames.index[frames.camera == c].to_numpy() for c in ("G299", "G330")}


def picks(run_id: str) -> list[int]:
    return [index[(a, b)] for a, b in m.sql(
        "SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ? ORDER BY rank", [run_id]).fetchall()]


def describe(rows: list[int]) -> dict:
    sub = frames.loc[rows]
    h = np.concatenate([heights.get(key[i], []) for i in rows] or [[]])
    cover = []
    for cam, members in cam_rows.items():
        chosen = [i for i in rows if frames.camera[i] == cam]
        x, c = vectors[frames.row[members].to_numpy()], vectors[frames.row[chosen].to_numpy()]
        d = np.full(len(members), np.inf)
        for k in range(0, len(c), 200):
            d = np.minimum(d, np.sqrt(((x[:, None, :] - c[None, k:k + 200, :]) ** 2).sum(-1)).min(1))
        cover.append(d)
    return {
        "G299": (sub.camera == "G299").mean(),
        "people mean": sub.people.mean(), "people p10-p90": f"{sub.people.quantile(.1):.0f}-{sub.people.quantile(.9):.0f}",
        "box h median": np.median(h), "boxes < 40px": (h < 40).mean(),
        "minutes": sub.groupby("camera").minute.nunique().sum(),
        "coverage": np.concatenate(cover).mean(),
        "radius": np.concatenate(cover).max(),
    }


print(f"pool: {len(frames)} candidates, minutes {frames.groupby('camera').minute.nunique().sum()}")
print("all candidates:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in describe(list(range(len(frames)))).items()})
for n in BUDGETS:
    print(f"\n--- first {n} (seed 0)")
    for method, runs in SELECTIONS.items():
        d = describe(picks(runs[0])[:n])
        print(f"{method:18s}", "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items()))
print("\nshared frames at 1,000 (seed 0 across methods; seeds 0/1/2 within a method):")
sets = {m_: [set(picks(r)) for r in runs] for m_, runs in SELECTIONS.items()}
names = list(SELECTIONS)
for i in range(3):
    for j in range(i + 1, 3):
        print(f"  {names[i]} ∩ {names[j]}: {len(sets[names[i]][0] & sets[names[j]][0])}")
for m_, s in sets.items():
    print(f"  {m_} seeds: 0∩1 {len(s[0] & s[1])}, 0∩2 {len(s[0] & s[2])}, 1∩2 {len(s[1] & s[2])}")
