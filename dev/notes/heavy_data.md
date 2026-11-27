# Heavy data

Everything heavy and local lives in `workspace/` (gitignored, D-21).

```mermaid
flowchart TB
    W["workspace/ (gitignored)"] --> MED["media/"]
    W --> CK["checkpoints/"]
    W -.-> ST["store/ (from M1: runs, judgments, features)"]
    W --> BK["backups/ (human runs, dev/scripts/backup_human_runs.sh)"]
    MED --> MEVA["meva/pool · meva/val · meva/test"]
    MED --> SMP["samples/ (quick-check images and videos)"]
    CK --> C1["sam3/ · yolo/"]
```

## MEVA slice (holy grail pool and test)

To re-download: `bash dev/scripts/pull_meva.sh` (needs only curl; skips files already on disk).

| Set | Path | Content |
|---|---|---|
| Pool | `workspace/media/meva/pool/` | School site, cameras G299 and G330, 2018-03-07 09:55–10:55: 24 clips of 5 min, 4.2 GB |
| Val | `workspace/media/meva/val/` | Same cameras, the pool day, 11:30–11:40 (D-31): 4 clips of 5 min |
| Test | `workspace/media/meva/test/` | Same cameras, five other days (D-32), 15 min per camera each: 2018-03-09 10:10, 03-11 11:15, 03-12 10:00, 03-13 15:55, 03-15 14:55. 30 clips |

- **Format:** H.264 in `.avi` with B-frames, 1920×1080. Measured at ingest (M1): 30 clips, 270,243 frames; every gap between frames is exactly 1/30 s; each clip's frame count matches ffprobe and the AVI header. The pool sampled at 2 fps is 14,424 frames.
- **Why this slice:** one site and two fixed cameras over a continuous hour is what raw CCTV looks like. The test footage comes from other days, so it shares no near-duplicate frames with the pool (D-13), and from five of them, so no single event dominates it (D-32).
- **Other hours on the bucket** for G299 and G330 (checked 2026-11-26, clips per hour folder, both cameras): 03-05 at 10 and 11 h; 03-07 at 09, 11 (val, D-31) and 17 h (17:00–17:40; an empty gym at 17:00); 03-11 at 11, 14 and 16 h; 03-12 at 10 h; 03-13 at 16 and 17 h; 03-15 at 15 h. 03-09 has only hour 10.
- **Source:** the public S3 bucket `mevadata-public-01` ([AWS Open Data](https://registry.opendata.aws/mevadata/)), folder `drops-123-r13/`. Paper: [arXiv 2012.00914](https://arxiv.org/abs/2012.00914).
- **Activity annotations** (37 types, for the actions work later) are not downloaded yet. Kitware publishes them separately.
- **Licence:** CC BY 4.0. Attribution required: *"Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA)*, licensed under CC BY 4.0.

## Samples

`workspace/media/samples/`: small media for quick checks and demos (smoke tests of a servant, not experiments).

| Kind | Files |
|---|---|
| Images (11) | `aquatic`, `crowd`, `dishes`, `girl`, `groceries`, `human`, `kitchen`, `merchandise`, `merchandise2`, `trash`, `wildlife` (`.jpg`) |
| Videos (8) | `01_dog`, `02_cups`, `03_blocks`, `04_coffee` (720p, 29.97 fps, about 10 s); `05_juggle` (720p, 24 fps, 10 s); `30s_vid` (720p, 30 fps, 35 s); `demo_skeleton` (1080p, 30 fps, 2.6 s); `vestTrainingVideo` (1799×908, 29.33 fps, 49 s) |

## Weights

```mermaid
flowchart LR
    subgraph LOCAL["workspace/checkpoints/ (plain files)"]
      SAM["sam3/<br/>sam3.pt · sam3.1_multiplex.pt · BPE vocab"]
      YOLO["yolo/<br/>yolo26* · yolov8* · yolo11n · rtdetr-l"]
    end
    subgraph HF["~/.cache/huggingface/hub (loaded with from_pretrained)"]
      D3["DINOv3 ViT-L/16 · ViT-H+/16"]
      SP["Sapiens2 1B: pretrain · pose · seg"]
      EG["EmbeddingGemma 2"]
      QW["Qwen3-VL 4B · 8B FP8"]
      S3["facebook/sam3"]
    end
```

- **Local files** are for libraries that take a file path (the SAM3 engine, Ultralytics).
- **Everything that loads with `from_pretrained`** stays in the HF cache: don't copy it into `workspace/`.
- v1's tuned `botsort_best.yaml` was left behind on purpose, because its tuning was broken (v1 CR-29). Sapiens 1 checkpoints stay in `dev/rin_v1`, unused.
