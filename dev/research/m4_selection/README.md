# What do the three selection methods pick from the pool?

**Question.** Before any student is trained: do `kcenter` and `kmeans_stratified` pick frames that differ from random's, in what the frames show and in DINOv3's sense of "different"? If curated picks look like random picks, the gym may be too uniform for whole-frame vectors to matter.

**Setup** (M4 steps 1, 3 and 6c, 2026-11-27).
- **The pool** (D-41): the pool day's busy 09:55–10:55 and quiet 17:10–17:40, cameras G299 and G330. **The chain** ([pool_chain.py](pool_chain.py)): sample 2 fps (21,636 frames, `20261127-205124-curate.union-81f9`) → DINOv3 ViT-L/16 CLS at 896×512, commit `ea8dc28` (`20261127-205124-dinov3.embed-a983`) → dedupe 0.99 (`20261127-210554-curate.dedupe-6a1d`): **4,683 candidates**, no motion filter. The busy hour gives the same 4,453 candidates as M4's motion-filtered chain (that filter dropped only 226 frames, and dedupe drops them anyway); the quiet half-hour gives **230**. Val (D-31), the same chain on 11:30–11:40 and 17:00–17:10: 321 candidates, 13 of them quiet (`20261127-210936-curate.dedupe-7cf2`). DINOv3 on all 26,444 pool and val frames: about 18 min.
- **Selections** (`curate.select`, D-38: per camera, cameras in turns; kcenter and kmeans_stratified on Euclidean distance between raw CLS vectors), `n_max` 1,000:

| Method | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| random | `20261127-211121-curate.select-8f6c` | `20261127-211121-curate.select-a594` | `20261127-211121-curate.select-ba0f` |
| kcenter | `20261127-211121-curate.select-49c7` | `20261127-211130-curate.select-f598` | `20261127-211139-curate.select-ec4e` |
| kmeans_stratified (50 per camera) | `20261127-211148-curate.select-f500` | `20261127-211158-curate.select-a364` | `20261127-211207-curate.select-1516` |

- **Measured** by [measure.py](measure.py), at seed 0: frames from the quiet footage; people per frame (SAM3 ≥ 0.5, [m4_training_labels](../m4_training_labels/README.md)); camera-minutes covered; and in DINOv3's space, for every candidate the distance to its nearest picked frame of the same camera, as a mean (typical coverage) and a maximum (the radius kcenter minimises). Contact sheets by [sheets.py](sheets.py).

## Results

All candidates: 47.5 people per frame (p10–p90 25–60), 136 camera-minutes; 230 quiet, 212 with at most 5 people, **3 with nobody** (SAM3 ≥ 0.5). Dedupe shrinks a stretch of empty gym to about one frame per camera; what the quiet half-hour adds is mostly an open floor with 1–9 people crossing ([quiet_sample.jpg](quiet_sample.jpg): SAM3's boxes there sit on real people).

| First N | Method | Quiet frames | ≤ 5 people | People per frame (p10–p90) | Camera-minutes | Mean distance to nearest pick | Radius |
|---|---|---|---|---|---|---|---|
| 25 | random | 0 | 0 | 51.0 (44–60) | 22 | 4.21 | 8.56 |
| 25 | kcenter | 2 | 3 | 44.5 (10–58) | 23 | 4.71 | **8.00** |
| 25 | kmeans_stratified | 3 | 1 | 45.4 (21–58) | 24 | 4.07 | 8.80 |
| 100 | random | 3 | 3 | 48.5 (30–59) | 69 | 3.41 | 8.56 |
| 100 | kcenter | 7 | 7 | 45.9 (20–62) | 54 | 4.14 | **5.88** |
| 100 | kmeans_stratified | 7 | 8 | 45.1 (14–60) | 68 | 3.38 | 8.42 |
| 250 | random | 10 | 8 | 48.8 (37–59) | 102 | 3.01 | 8.52 |
| 250 | kcenter | 13 | 16 | 46.5 (21–61) | 90 | 3.55 | **4.84** |
| 250 | kmeans_stratified | **19** | 17 | 45.6 (14–60) | 97 | 3.01 | 8.42 |
| 1,000 | random | 36 | 27 | 48.6 (31–60) | 123 | 2.16 | 8.52 |
| 1,000 | kcenter | 39 | 40 | 48.4 (28–60) | 120 | 2.21 | **3.43** |
| 1,000 | kmeans_stratified | **81** | 75 | 45.1 (12–60) | 125 | 2.18 | 8.27 |

Every method picks at most one of the 3 empty frames at any budget. Box heights barely differ (median 91–94 px, 3.0–3.7% under 40 px). Frames shared at 1,000: random ∩ kcenter 204, random ∩ kmeans 215, kcenter ∩ kmeans 284. Between seeds: random about 215, kmeans 246–283, **kcenter 740–792**.

First 50 picks of each: [random](first50_random.jpg), [kcenter](first50_kcenter.jpg), [kmeans_stratified](first50_kmeans_stratified.jpg).

## Verdict

- **With a quiet scene in the pool, both curated methods reach for it.** At 250 frames random holds 10 quiet frames, kcenter 13, kmeans_stratified 19; at 1,000, kmeans holds 81 against random's 36. In M4's busy-only pool kmeans_stratified couldn't be told from random; with two kinds of scene, its clusters give the small one more than its 5% share.
- **kcenter still does what it promises:** it shrinks the worst-case distance at every budget (4.84 against random's 8.52 at 250), at the cost of a worse typical distance, and reaches the sparse frames earliest (3 of its first 25 hold at most 5 people; random's hold none).
- **No method brings empty frames:** the pool holds 3. Sparse frames (an open floor, few people) are what the quiet footage adds; whether they stop the students' boxes on empty test frames is M5's to measure.
- **kcenter's seed only moves its first frame,** so its seeds share about 77% of their frames: its spread across seeds in M5 will be mostly training noise.
