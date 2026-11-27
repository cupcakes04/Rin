# Does a student learn from SAM3's unchecked labels, and does it improve with more frames?

**Question.** Before M5 trains 36 students: does yolo26s trained on SAM3's unchecked pool labels beat stock yolo26s on the checked test set, and does test mAP rise from 100 to 1,000 frames? If the curve is flat, selection can't show a difference and M5's budgets must change.

**Setup** (M4 steps 4 and 5, 2026-11-27).
- **Training set:** random's ranking at seed 0 (`20261127-132314-curate.select-5da9`), the first N frames, with D-39's labels; val: the 308 val candidates, same labels (D-31). Export `20261127-150150-export.yolo-55bf` (N = 1,000) holds 49,462 train boxes and 16,434 val boxes; boxes drawn back from its label files sit on the people ([export_check.jpg](export_check.jpg)).
- **Student** (D-40): yolo26s from the stock COCO checkpoint, Ultralytics 8.4.220, `imgsz` 1280, batch 8, 50 epochs, seed 0, no early stop; `best.pt` picked on val.
- **Scoring:** `yolo.predict` on the 300 checked test frames (confidence 0.001, at most 100 boxes per frame; no frame holds more than 60 people), `evaluate.coco` against `export.coco` run `20261127-152506-export.coco-8bc8` (pycocotools; crowd regions ignored, D-34). Scripts: [timing.py](timing.py), [curve.py](curve.py).

## Results

**Image size** (10 epochs on 1,000 frames; val mAP is against SAM3's unchecked val labels, so it shows learning, not accuracy):

| Run | imgsz · batch | Time per epoch | Peak VRAM | Val mAP50-95 after epochs 1 / 5 / 10 |
|---|---|---|---|---|
| `20261127-150308-yolo.train-640e` | 640 · 16 | 9.8 s | 4.5 GiB | 0.36 / 0.49 / 0.54 |
| `20261127-150503-yolo.train-75f2` | 1280 · 8 | 26.0 s | 9.3 GiB | 0.61 / 0.65 / 0.69 |

A 1-epoch run (`20261127-150221-yolo.train-aab1`) scores 0.002: an artefact of a 1-epoch schedule (epoch 1 of a 10-epoch run scores 0.36).

**The curve** (test mAP on the checked test set):

| N | Train run | mAP50-95 | mAP50 | mAP_small | AR100 | False positives per empty frame (≥ 0.5) | Training |
|---|---|---|---|---|---|---|---|
| 0 (stock, 640) | — | 0.418 | 0.667 | 0.008 | 0.49 | 0.00 | — |
| 0 (stock, 1280) | — | 0.599 | 0.828 | 0.090 | 0.66 | 0.00 | — |
| 100 | `20261127-152532-yolo.train-4177` | 0.619 | 0.843 | 0.178 | 0.72 | 3.22 | 5.7 min |
| 250 | `20261127-153127-yolo.train-61a5` | 0.646 | 0.850 | 0.275 | 0.73 | 2.95 | 8.7 min |
| 500 | `20261127-154021-yolo.train-fabe` | 0.651 | 0.864 | 0.261 | 0.73 | 1.73 | 12.8 min |
| 1,000 | `20261127-155322-yolo.train-1dac` | 0.661 | 0.882 | 0.277 | 0.73 | 0.30 | 22.5 min |

Evaluation runs: stock `20261127-152517-evaluate.coco-037b` (640), `20261127-152529-evaluate.coco-7d92` (1280); students `20261127-153124-evaluate.coco-1abc`, `20261127-154017-evaluate.coco-2a24`, `20261127-155315-evaluate.coco-53d9`, `20261127-161556-evaluate.coco-dedd`. Peak VRAM 10.0 GiB; training time is about 21 ms per frame and epoch plus a fixed part.

**Known faults, redone in M4 step 6** ([audit 2026-11-27](../../audits/2026-11-27.md), triage at its end): the stock rows are too low, because `yolo.predict` kept persons only after Ultralytics' 100 boxes per frame were shared among all 80 COCO classes (44 person boxes on the 60-person frame); and the fixed 50 epochs gave small budgets far fewer optimizer steps (about 80 at 100 frames, 780 at 1,000), so the gains below and the fall of ghosts with N mix frames with training. No random budget here holds an empty frame.

## Verdict

- **The student learns from unchecked labels**, and beats stock yolo26s at every budget, most of all on **small people**: mAP_small 0.09 → 0.28, AR_small 0.23 → 0.42–0.48. Domain tuning helps where a stock detector is weakest, which is the point of the loop.
- **The curve rises, but flattens early:** +0.020 from stock to 100 frames, +0.027 to 250, then +0.005 to 500 and +0.010 to 1,000. One seed, and test frames that cluster in time, so 1–2 points aren't readable as real. D-17's bar (random's mAP with half the labels) can only be decided where the curve still climbs, **below about 250 frames**: M5's budgets should move down (e.g. 25 / 50 / 100 / 250 / 500), which is also cheaper.
- **Small students box ghosts on empty frames.** Stock puts no box at ≥ 0.5 on the 110 empty test frames; the 100-frame student puts 3.2 per frame, falling to 0.3 at 1,000. The candidate pool holds no empty gym (a busy hour, and the motion filter drops the few empty frames), so a small training set never shows one; some of SAM3's training boxes are objects. A method that picks the rare near-empty frames (kcenter does, [m4_selection](../m4_selection/README.md)) may win here first, so M5 reports this number per method.
