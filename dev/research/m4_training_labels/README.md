# Which of SAM3's boxes should train the student?

**Question.** The student learns from SAM3's unchecked boxes (usage C2). Which ones: above what score, of what size, and are the "same spot all window" boxes that were bags and coats on the test days (M3) also errors in the pool? The answer is D-39.

**Setup** (M4 step 2, 2026-11-27).
- **SAM3 on every candidate** (D-37): run `20261127-111139-sam3.concept-5477` on `20261127-111139-curate.union-365e`, the 4,453 pool candidates (`20261127-105927-curate.dedupe-6b65`) and 308 val candidates (`20261127-110156-curate.dedupe-f05d`); script [sam3_all.py](sam3_all.py).
- **The gate** is measured on test frames against two truths: the checked test set (`20261127-094340-human.annotate-64f9`, 300 frames) and the blind pilot (`20261126-151047-human.annotate-3bd2`, 18 frames, 180 people), with SAM3 run `20261126-205940-sam3.concept-11ee`. Script [gate.py](gate.py): a SAM3 box is right when it matches a checked box at IoU ≥ 0.5 (greedy, highest score first); an unmatched box at least half inside a `crowd` box is ignored (D-34).
- **The bags:** `human.suspects(above=0.5)` on the pool candidates' labels.

## Results

**Cost.** 675,492 detections above 0.1 on 4,761 frames in 6,491 s: **1.36 s per frame** (0.78 on the sparser test frames), peak VRAM 6.6 GiB. Above 0.5 on the pool: 220,436 boxes, about 49 per frame.

**The gate:**

| Gate | Test set: right / wrong | Test set: precision · recall | Blind pilot: precision · recall · F1 |
|---|---|---|---|
| 0.30 | 3,429 / 1,436 | 0.705 · 0.999 | 0.689 · 0.922 · 0.789 |
| 0.40 | 3,429 / 785 | 0.814 · 0.999 | 0.756 · 0.911 · 0.826 |
| 0.45 | 3,429 / 533 | 0.865 · 0.999 | 0.770 · 0.894 · 0.828 |
| **0.50** | **3,429 / 259** | **0.930 · 0.999** | **0.808 · 0.889 · 0.847** |
| 0.55 | 3,218 / 193 | 0.943 · 0.938 | 0.819 · 0.878 · 0.847 |
| 0.60 | 2,998 / 128 | 0.959 · 0.874 | 0.852 · 0.867 · 0.860 |
| 0.70 | 2,528 / 48 | 0.981 · 0.737 | 0.923 · 0.800 · 0.857 |

- **The test set's recall is anchored at 0.5.** Its check started from SAM3's boxes above 0.5 (D-33) and added almost none of the 0.3–0.5 hints, so below 0.5 it can't show missed people. The blind pilot can: lowering the gate from 0.5 to 0.3 finds 6 more people (3%) for 37 more wrong boxes. Above 0.5 the test set is honest: people the checker kept drop out quickly.
- **0.5 is the knee:** from 0.45 to 0.5 the wrong boxes halve and no checked person is lost. Choosing the gate on the test set favours no method: every method gets the same gate.
- **No size gate:** 79 of the 259 wrong boxes at 0.5 are under 40 px tall, but so are 340 of the 3,432 checked people (all 20–40 px; none smaller).

**The bags.** 15,355 pool boxes (7%) sit on the same spot in at least half of their camera's frames, in 45 spots on 4,151 of 4,453 frames; 626 more are near-duplicates. The 30 largest spots, cropped at the start, middle and end of the hour ([pool_same_spot.jpg](pool_same_spot.jpg), one row per spot), are almost all **people seated on the bleachers**: one seat, occupied all hour. Close-ups of the odd ones ([pool_same_spot_zoom.jpg](pool_same_spot_zoom.jpg)): people standing at the screen's edge, a person lying on a bench, and one real error of another kind, a box over two seated people (773 boxes). No bag or coat; the user looked and agreed. Checking the pool needed a fix to `human.suspects`: its same-spot check built a 90 GiB box-by-box matrix and now works in chunks, with identical flags on the test set.

## Verdict

D-39: training labels are SAM3's boxes at **score ≥ 0.5, no size gate, same-spot boxes kept**, and of two near-duplicates (IoU ≥ 0.7) the higher-scored one. In a busy hour "same spot" means a seat, not an object: a rule tuned on sparse windows doesn't carry over to busy ones. What stays in training is teacher noise: about 7% wrong boxes at 0.5 (on test days, outside crowds), and boxes over two people, which no rule here catches.
