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
| SAM3 / SAM3.1 | Spatial teacher: concept prompts, masks, video tracking | `vendors/sam3_engine` (the user's engine) | `sam3.pt`, `sam3.1_multiplex.pt`, BPE vocab, in `workspace/checkpoints/sam3/` (6.5 GB) | worked in v1 |
| DINOv3 | Features: curation, de-duplication, search, action-head input | HF `transformers` ([collection](https://huggingface.co/collections/facebook/dinov3)) | HF cache | worked in v1; variant for Rin not chosen |
| EmbeddingGemma 2 | Text ↔ image/video embeddings, search | `vendors/gemma_emb_engine` (the user's engine) | HF, pinned revision | worked in v1 |
| **YOLO** | **Holy grail student** (D-17), edge tier. Tasks: detect, segment, pose, [semantic segmentation](https://docs.ultralytics.com/tasks/semantic) (for stuff classes, D-06) | `ultralytics` package | stock: `yolo26*`, `yolov8*`, `yolo11n` in `workspace/checkpoints/yolo/` | worked in v1; semantic task untested |
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
- **SAM3 object limits, for crowded CCTV (about 30 people per scene):** per-image detection has 200 queries (`num_queries=200` in the vendored `model_builder.py`), so a crowded frame fits; the holy grail only uses per-frame detection. Tracking is different: SAM 3.1 multiplex tracks at most `EngineConfig.multiplex_max_num_objects` objects (default 16), and both video sessions use more VRAM per tracked object. Matters for [tracker tuning](../ideas/tracker_tuning.md); not measured here yet.
- **vLLM on WSL2:** see [vlm.md](vlm.md).

## v1 notes worth reading before writing a servant

`dev/rin_v1/docs/notes/services/`: `sam3/` (image and video usage), `gg_emb2/`, `qwen/` (install, prompting, 4B and 8B FP8 fits), `sapiens/install.md`, `boxmot/install.md`.

Working environment pins per model (Python version, CUDA wheel index, BoxMOT extras `[cu130,evolve,service]`, `torchcodec` for EmbeddingGemma video): `dev/rin_v1/src/rin/cli/configs/environments.yaml`. A starting point for each servant's `uv` environment, not a copy.
