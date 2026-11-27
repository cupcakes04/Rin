# What do the three selection methods pick from the pool?

**Question.** Before any student is trained: do `kcenter` and `kmeans_stratified` pick frames that differ from random's, in what the frames show and in DINOv3's sense of "different"? If curated picks look like random picks, the gym may be too uniform for whole-frame vectors to matter (the testbed doubt, parked for M5's verdict).

**Setup** (M4 steps 1 and 3, 2026-11-27).
- **The pool chain, rebuilt in `workspace/store`** ([pool_chain.py](pool_chain.py)): sample 2 fps `20261127-104506-core.sample-b156` (14,424) → keep `motion >= 0.002` `20261127-104644-curate.keep-7b34` (14,198) → DINOv3 ViT-L/16 CLS at 896×512, commit `ea8dc28` (`20261127-104644-dinov3.embed-32fd`) → dedupe 0.99 `20261127-105927-curate.dedupe-6b65`: **4,453 candidates**, the same frames as M2's, compared by file and frame index ([m2_filters_dedupe](../m2_filters_dedupe/README.md)). Val, 03-07 11:30–11:40 (D-31), the same chain: 308 candidates.
- **Selections** (`curate.select`, D-38: per camera, Euclidean on raw CLS, cameras in turns), `n_max` 1,000:

| Method | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| random | `20261127-132314-curate.select-5da9` | `20261127-132314-curate.select-8594` | `20261127-132314-curate.select-4375` |
| kcenter | `20261127-132314-curate.select-8f97` | `20261127-132326-curate.select-5543` | `20261127-132338-curate.select-9aab` |
| kmeans_stratified (50 per camera) | `20261127-132349-curate.select-2d06` | `20261127-132359-curate.select-d369` | `20261127-132408-curate.select-c75a` |

- **Measured** by [measure.py](measure.py), at seed 0: people per frame (SAM3 ≥ 0.5, [m4_training_labels](../m4_training_labels/README.md)); camera-minutes covered; and in DINOv3's space, for every candidate the distance to its nearest picked frame of the same camera, as a mean (typical coverage) and a maximum (the radius kcenter minimises).

## Results

All candidates: 49.7 people per frame (p10–p90 38–60), 118 camera-minutes.

| First N | Method | People per frame (p10–p90) | Camera-minutes | Mean distance to nearest pick | Radius |
|---|---|---|---|---|---|
| 100 | random | 50.2 (39–59) | 62 | 3.37 | 8.62 |
| 100 | kcenter | 47.0 (**22**–62) | 54 | 4.10 | **5.71** |
| 100 | kmeans_stratified | 49.0 (37–60) | 66 | 3.35 | 8.54 |
| 250 | random | 49.6 (37–60) | 95 | 2.99 | 8.42 |
| 250 | kcenter | 48.9 (30–61) | 81 | 3.49 | **4.69** |
| 250 | kmeans_stratified | 48.8 (34–60) | 90 | 2.96 | 7.87 |
| 1,000 | random | 49.5 (38–60) | 115 | 2.12 | 6.87 |
| 1,000 | kcenter | 50.0 (38–61) | 109 | 2.16 | **3.36** |
| 1,000 | kmeans_stratified | 49.1 (34–60) | 113 | 2.11 | 7.37 |

Box heights barely differ (median 90–93 px, 3.3–3.8% under 40 px). Frames shared at 1,000: random ∩ kcenter 234, random ∩ kmeans 226, kcenter ∩ kmeans 286. Between seeds: random about 230, kmeans 238–277, **kcenter 766–778**.

First 50 picks of each: [random](first50_random.jpg), [kcenter](first50_kcenter.jpg), [kmeans_stratified](first50_kmeans_stratified.jpg).

## Verdict

- **kcenter does what it promises, and loves outliers.** It halves the worst-case distance at every budget, at the cost of a worse typical distance, and its first picks are the hour's extremes: an almost empty gym (1–2 people), people right in front of the camera, a cone, a light's flare. Random's are typical 45–60-person frames, with no empty one (the user's look agrees).
- **kmeans_stratified can't be told from random** on any number here, or by eye. In a uniform hour, 50 clusters per camera spread the picks about as chance does. This is the early warning about the gym: whole-frame vectors find little variety to exploit, except at the extremes.
- **kcenter's seed only moves its first frame,** so its seeds share 77% of their frames: its spread across seeds in M5 will be mostly training noise.
- Which kind of frame teaches the student more is M5's question; [m4_first_curve](../m4_first_curve/README.md) suggests where to look (empty frames, low budgets).
