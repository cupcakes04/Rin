# Does a student learn from SAM3's unchecked labels, and does it improve with more frames?

**Question.** Before M5 trains many students: does yolo26s trained on SAM3's unchecked pool labels beat stock yolo26s on the checked test set, does test mAP rise with more frames, and how hard should a student train? If the curve is flat, selection can't show a difference and M5's budgets or metrics must change.

**Setup** (M4 steps 4, 5 and 6b, 2026-11-27).
- **Training set:** a random ranking of the pool candidates, the first N frames, with D-39's labels; val: the 308 val candidates, same labels (D-31). Export `20261127-150150-export.yolo-55bf` (N = 1,000) holds 49,462 train boxes and 16,434 val boxes; boxes drawn back from its label files sit on the people ([export_check.jpg](export_check.jpg)).
- **Student** (D-40): yolo26s from the stock COCO checkpoint, Ultralytics 8.4.220, `imgsz` 1280, batch 8, 50 epochs, seed 0, no early stop, AdamW at 0.002 with one optimizer step per 8 batches; `best.pt` picked on val.
- **Scoring:** `yolo.predict` on the 300 checked test frames (confidence 0.001, at most 100 person boxes per frame, other classes dropped before the cut), `evaluate.coco` against `export.coco` run `20261127-152506-export.coco-8bc8` (pycocotools; crowd regions ignored, D-34). Scripts: [timing.py](timing.py), [curve.py](curve.py), [steps.py](steps.py).

## Results

**Image size** (10 epochs on 1,000 frames; val mAP is against SAM3's unchecked val labels, so it shows learning, not accuracy):

| Run | imgsz · batch | Time per epoch | Peak VRAM | Val mAP50-95 after epochs 1 / 5 / 10 |
|---|---|---|---|---|
| `20261127-150308-yolo.train-640e` | 640 · 16 | 9.8 s | 4.5 GiB | 0.36 / 0.49 / 0.54 |
| `20261127-150503-yolo.train-75f2` | 1280 · 8 | 26.0 s | 9.3 GiB | 0.61 / 0.65 / 0.69 |

A 1-epoch run (`20261127-150221-yolo.train-aab1`) scores 0.002: an artefact of a 1-epoch schedule (epoch 1 of a 10-epoch run scores 0.36).

**The curve** (test mAP on the checked test set; random seed 0 over all candidates, before D-38 made random take the cameras in turns):

| N | Train run | mAP50-95 | mAP50 | mAP75 | mAP_small | AR100 | False positives per empty frame (≥ 0.5) | Training |
|---|---|---|---|---|---|---|---|---|
| 0 (stock) | — | 0.629 | 0.885 | 0.702 | 0.167 | 0.71 | 0.00 | — |
| 100 | `20261127-152532-yolo.train-4177` | 0.619 | 0.843 | 0.686 | 0.178 | 0.72 | 3.22 | 5.7 min |
| 250 | `20261127-153127-yolo.train-61a5` | 0.646 | 0.850 | 0.712 | 0.275 | 0.73 | 2.95 | 8.7 min |
| 500 | `20261127-154021-yolo.train-fabe` | 0.651 | 0.864 | 0.717 | 0.261 | 0.73 | 1.73 | 12.8 min |
| 1,000 | `20261127-155322-yolo.train-1dac` | 0.661 | 0.882 | 0.728 | 0.277 | 0.73 | 0.30 | 22.5 min |

Stock: predictions `20261127-175045-yolo.predict-c01e`, scored `20261127-194939-evaluate.coco-f23f`. Students: `20261127-153124-evaluate.coco-1abc`, `20261127-154017-evaluate.coco-2a24`, `20261127-155315-evaluate.coco-53d9`, `20261127-161556-evaluate.coco-dedd`. Peak VRAM 10.0 GiB; training time is about 21 ms per frame and epoch plus about 4.5 s per epoch, mostly the val pass.

**How hard to train** (step 6b; per-camera random, seed 0, selection `20261127-180448-curate.select-e91e`; "steps" are optimizer steps). M4's setting is one step per 8 batches; the others take one step per batch, some with each frame repeated so an epoch shows 250 or 1,000 images (the same 50 epochs and val passes):

| N | Setting | Steps | mAP50-95 | mAP50 | mAP_small | AR100 | False positives per empty frame | Train run |
|---|---|---|---|---|---|---|---|---|
| 25 | M4's | 25 | 0.634 | 0.889 | 0.174 | 0.71 | 0.00 | `20261127-183232-yolo.train-468c` |
| 25 | step per batch | 200 | 0.557 | 0.806 | 0.175 | 0.66 | 1.71 | `20261127-180450-yolo.train-57a5` |
| 25 | per batch, 250 per epoch | 1,600 | 0.531 | 0.751 | 0.040 | 0.62 | 5.29 | `20261127-181352-yolo.train-9e7c` |
| 25 | per batch, 1,000 per epoch | 6,250 | 0.517 | 0.734 | 0.037 | 0.61 | 5.87 | `20261127-184148-yolo.train-f64f` |
| 100 | M4's | 81 | 0.620 | 0.852 | 0.232 | 0.71 | 1.92 | `20261127-183624-yolo.train-f9f0` |
| 100 | step per batch | 650 | 0.608 | 0.839 | 0.202 | 0.70 | 1.82 | `20261127-180841-yolo.train-369f` |
| 100 | per batch, 250 per epoch | 1,900 | 0.603 | 0.830 | 0.191 | 0.68 | 1.65 | `20261127-182151-yolo.train-94c3` |
| 100 | per batch, 1,000 per epoch | 6,250 | 0.570 | 0.810 | 0.102 | 0.65 | 2.05 | `20261127-190408-yolo.train-0eef` |
| 1,000 | step per batch | 6,250 | 0.630 | 0.843 | 0.241 | 0.71 | 2.19 | `20261127-192636-yolo.train-37db` |

The repeat option (`yolo.train(epoch_images=...)`) was removed after this measurement; [steps.py](steps.py) needs it to run again (git history has it).

**SAM3 on the same test set** (run `20261126-205940-sam3.concept-11ee`, scored `20261127-194940-evaluate.coco-d670`): mAP50-95 0.966, mAP_small 0.846, 1.3 false positives per empty frame. Not a ceiling for the student: the checked truth started as SAM3's boxes and the person kept most of them, so SAM3 matches it almost by construction. What it does show: the truth carries SAM3's box geometry, and a student trained on SAM3's boxes is rewarded at high IoU for learning it.

## Verdict

- **Stock yolo26s is already strong here:** 0.629 mAP50-95, the best mAP50 of any model (0.885), and no box on an empty frame. Fine-tuning on unchecked labels gains at most **+0.03 mAP50-95** (1,000 frames), never beats stock on mAP50, and adds boxes on empty frames at most budgets.
- **Where students do gain is small people:** mAP_small 0.17 → 0.26–0.28 from 250 frames on. Their mAP75 rises too (0.70 → 0.73): part of the overall gain is learning SAM3's box shape, which the truth inherits.
- **Train gently.** More optimizer steps made students worse at every budget tried: fewer small people found, more boxes on empty frames, lower mAP. The audit's worry (2026-11-27 §1.1) that 50 epochs starved small budgets is measured and rejected; D-40 keeps M4's setting, now stated.
- **Boxes on empty frames come from training, not from too little of it.** None of these training sets holds an empty frame; the student that trains least (25 frames, 25 steps) puts none, like stock. Quiet frames in the pool (D-41, step 6c) are the direct test.
- **The curve is too flat and too noisy for D-17 on overall mAP.** One seed: 25 frames (0.634) beat 100 (0.619, 0.620 on two different draws), and the whole range from stock to 1,000 frames is 0.03. "Half the labels" can't be read on mAP50-95 here; mAP_small and false positives per empty frame are the metrics with room to move ([audit 2026-11-27 §4.1](../../audits/2026-11-27.md)).
