# Models

What each model is for in Rin, where it comes from, and what its model card doesn't say. A model gets its own file here only when it has enough notes to need one (today: [vlm.md](vlm.md)).

**Rules:**
- A servant calls the vendor's own API directly (D-01, D-18).
- Before writing one, read the model card and the vendor README.
- Record here only what they leave out. Never copy a model card: link it.

## Roles at a glance

```mermaid
flowchart LR
    subgraph T["Teachers (slow, accurate)"]
      SAM3; SAP["Sapiens2"]; VLM["Qwen3-VL / Gemma 4"]
    end
    subgraph F["Features"]
      DINO["DINOv3"]; EMB["EmbeddingGemma 2"]
    end
    subgraph S["Students (fast, deployable)"]
      YOLO["YOLO ★ first"]; RFD["RF-DETR (fallback)"]
    end
    TRK["BoxMOT"]
    T --> S
    F --> T
    S --> TRK
```

## Table

| Model | Role | Code | Weights | Status |
|---|---|---|---|---|
| SAM3 / SAM3.1 | Spatial teacher: concept prompts, masks, video tracking | `vendors/sam3_engine` (the user's engine) | `sam3.pt`, `sam3.1_multiplex.pt`, BPE vocab, in `workspace/checkpoints/sam3/` (6.5 GB) | in use (M3): `sam3.concept`, image model, 0.78 s/frame and 6.8 GiB peak at 8 images per session on the 300 test frames ([m3_sam3_test_day](../../research/m3_sam3_test_day/README.md)) |
| DINOv3 | Features: curation, de-duplication, search, action-head input | HF `transformers` ([collection](https://huggingface.co/collections/facebook/dinov3)) | HF cache | in use (M2): ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`), CLS at 896×512 |
| EmbeddingGemma 2 | Text ↔ image/video embeddings, search | `vendors/gemma_emb_engine` (the user's engine) | HF, pinned revision | worked in v1 |
| **YOLO** | **Holy grail student** (D-17), edge tier. Tasks: detect, segment, pose, [semantic segmentation](https://docs.ultralytics.com/tasks/semantic) (for stuff classes, D-06) | `ultralytics` package | stock: `yolo26*`, `yolov8*`, `yolo11n` in `workspace/checkpoints/yolo/` | in use (M4): `yolo.train`, `yolo.predict`, Ultralytics 8.4.220; yolo26s at 1280, batch 8: 26 s per epoch on 1,000 frames, 10 GiB ([m4_first_curve](../../research/m4_first_curve/README.md)); semantic task untested |
| RF-DETR | Student, GPU tier; fallback if YOLO hits a dead end | `rfdetr` package | stock from the package | worked in v1 |
| Sapiens2 | Person specialist: pose (308 keypoints), body-part segmentation (29 parts), actor features | Native in `transformers` ≥ 5.24 (`AutoModel` / `AutoBackbone` for features, `AutoModelForPoseEstimation` with person boxes, `AutoModelForSemanticSegmentation`); [docs](https://huggingface.co/docs/transformers/main/model_doc/sapiens2), [collection](https://huggingface.co/collections/facebook/sapiens2). No submodule: the 308 keypoint names come from upstream [keypoints308.py](https://github.com/facebookresearch/sapiens2/blob/main/sapiens/pose/configs/_base_/keypoints308.py) | HF cache: `facebook/sapiens2-{pretrain,pose,seg}-1b` (17.8 GB) | downloaded and verified, not run yet |
| BoxMOT | Tracker for edge, tuning | `vendors/boxmot` | ReID weights as needed. v1's tuned `botsort_best.yaml` is untrustworthy (tuning scored 0.0, v1 CR-29) | worked in v1; tuning broken |
| Qwen3-VL 4B / 8B FP8 | Actions (VLM teacher) | vLLM, own environment | HF | worked in v1 ([vlm.md](vlm.md)) |
| Gemma 4 | Actions (VLM teacher) | `transformers` or vLLM | HF | untested ([vlm.md](vlm.md)) |
| Mask2Former | Parked (D-06) | — | — | — |

## Environments

One environment per servant when dependencies clash, and they do: BoxMOT was on Python 3.11, EmbeddingGemma on 3.12, and vLLM needs its own. Managed with `uv`. The core library stays light and imports no model framework.

## Known gotchas

- **Sapiens2 weights come twice per HF repo:** `model.safetensors` (for `transformers`) and `sapiens2_<size>_<task>.safetensors` (for the vendor repo's standalone loader), about 6 GB each at 1B. Download only the first, naming the files: `hf download facebook/sapiens2-<task>-1b config.json preprocessor_config.json model.safetensors`. A multi-name `--include` silently dropped `config.json` here. Pose is top-down: pass person boxes as COCO `(x, y, w, h)` in pixels, and the processor crops. v1's Sapiens 1 `torch.export` problem doesn't apply.
- **DINOv3 input size:** the HF preprocessor squashes every image to 224×224 unless given `size={"height": h, "width": w}`. The model itself takes any size that is a multiple of 16 (rotary position embeddings, `rope_theta` in its config, are computed for whatever patch grid arrives; no fixed position table). Checked here on ViT-L/16 at 224×224, 448×256 and 1920×1088 (1080 padded to 1088): `last_hidden_state` has 1 + 4 + (h/16)(w/16) tokens. Rin's default is 896×512: a person 100 px tall at 1080p spans about 3 patches (1 at 224), at a sixth of native's cost. Speed here, ViT-L/16, float32 weights under float16 autocast, model only:

  | Input | Frames/s | Peak VRAM |
  |---|---|---|
  | 224×224 | ~315 | 1.6 GiB |
  | 448×256 | ~139 | 1.7 GiB |
  | 896×512 | ~32 (31.5 end to end from cached JPEGs) | 3.2 GiB |
  | 1920×1088 | ~5 | 2.3 GiB |
- **DINOv3 precision:** weights loaded as float16 overflow inside the network and every output is NaN (measured here on ViT-L/16: all 14,198 pool vectors). Load float32 weights and run under `torch.autocast("cuda", dtype=torch.float16)`: it matches float32 (cosine 1.0000) at about 3× float32's speed. That was v1's setting (`dino3/engine.py`); bfloat16 weights also work (cosine 0.9998), about 25% faster.
- **DINOv3 register tokens:** `last_hidden_state` is `[CLS, register tokens, patch tokens]`. Strip the registers before averaging patches or reshaping them into a 2-D grid, or the features are quietly wrong (mean) or the reshape fails. Read the count from `model.config.num_register_tokens`; v1 hard-coded positions `1:5` (4 registers) in `dev/rin_v1/src/rin/services/extract/dino3/engine.py` (`_clean_tokens`).
- **SAM3 (v1 CR-25):** a long-lived SAM3 process kept the previous job's prompts. Batch jobs (D-07) avoid this by design.
- **SAM3's tiny people are missing at the detector, not dropped later** (measured by the user on the 300 test frames, `sam3_engine` `bb14ef1`): 4 of 34,319 detections above 0.1 had an empty mask at full size, all scoring 0.5 or less. Since `bb14ef1`, `detect(keep_empty_mask_detections=...)` keeps or drops them (dropping logs the count; Rin drops), and each result carries two boxes: `mask_box_xyxy_px` (tight around the mask, what Rin stores) and `detector_box_xyxy_px` (SAM3's own regression, clipped to the image). Above 0.5 they agree (median IoU 0.86 at 20–40 px, 0.97 at 160 px and taller); below 0.5 they often don't (0.53–0.69 under 80 px). Recovering tiny people would need detection on zoomed-in tiles.
- **SAM3's stretched boxes come from split masks:** Rin stores the box around the whole mask, and when one detection's mask has two pieces (a stray patch on another person) the box spans both and the floor between. On the 300 test frames, 31 of the 43 boxes the user deleted had a gap of 20 px or more inside their mask, often scoring 0.8–0.94; 97 such boxes passed the first check (agent's count, audit [2026-11-26-m3](../../audits/2026-11-26-m3.md) §1.3). **SAM3's detector box doesn't have this fault** (run `20261126-205940-sam3.concept-11ee`, `box="detector"`; same 16,450 detections and masks as the deleted mask-box run `7efc`): where the mask is whole the two boxes agree (median IoU 0.92, 4,769 boxes above 0.5), where it is split (128) the detector box is about half the area (median IoU 0.45 with the mask box) and sat on one person in 16 of 16 looked at.
- **SAM3's main error on MEVA's gym is objects, not people:** bags, coats and chairs on the bleachers, boxed above 0.5 on the same spot in nearly every frame of a window (03-13 and 03-15 G330). On the 144 test frames with 0–5 SAM3 boxes, the person deleted 161 and drew none; above 0.5, 0.934 of SAM3's boxes are people outside crowds ([m3_checked_test_set](../../research/m3_checked_test_set/README.md)). On a fixed camera, a box on the same spot across many frames is suspect (`human.suspects`).
- **SAM3 object limits, for crowded CCTV:** per-image detection has 200 queries (`num_queries=200` in the vendored `model_builder.py`). On the test day's seated crowd, up to 194 detections above 0.1 per frame, but the floor still did the cutting (lowest score about 0.10), so the cap was not reached; the holy grail only uses per-frame detection. Tracking is different: SAM 3.1 multiplex tracks at most `EngineConfig.multiplex_max_num_objects` objects (default 16), and both video sessions use more VRAM per tracked object. Matters for [tracker tuning](../ideas/tracker_tuning.md); not measured here yet.
- **Ultralytics writes beside its inputs:** training saves a `labels.cache` next to the label folders it reads, so `yolo.train` trains on a hard-linked copy of the export (the export is a finished run). Its mixed-precision check loads `yolo26n.pt` from the working folder and downloads it if missing, so the job runs in the checkpoint folder, which holds it. A 1-epoch run scores near 0 (its learning-rate schedule never leaves warm-up): don't read a smoke test's mAP.
- **Ultralytics and `data.yaml`:** without a `path` key it resolves `train:` and `val:` beside the yaml file, so an export can move; a relative `path` would be resolved against Ultralytics' own datasets folder (v1 wrote an absolute one).
- **vLLM on WSL2:** see [vlm.md](vlm.md).

## v1 notes worth reading before writing a servant

`dev/rin_v1/docs/notes/services/`: `sam3/` (image and video usage), `gg_emb2/`, `qwen/` (install, prompting, 4B and 8B FP8 fits), `sapiens/install.md`, `boxmot/install.md`.

Working environment pins per model (Python version, CUDA wheel index, BoxMOT extras `[cu130,evolve,service]`, `torchcodec` for EmbeddingGemma video): `dev/rin_v1/src/rin/cli/configs/environments.yaml`. A starting point for each servant's `uv` environment, not a copy.
