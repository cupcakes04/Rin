# Brainstorm 2026-11-23: core reset (processed)

The raw text is in [raw_2026-11-23.md](raw_2026-11-23.md), unedited. This table records where each point went, so a later session doesn't need the chat.

| User's point | Agent's verdict | Where it lives now |
|---|---|---|
| v1 core was messy and bloated; redo it from scratch without v1's bias | Agreed. The schema was right; the layer between data and models was the problem | [lessons/rin_v1.md](../lessons/rin_v1.md), D-01, D-18 |
| Making all models modular under one schema backfires (instinct) | Correct. Standardise the data, not the models | [direction §2](../direction.md#2-the-core-principle), D-01 |
| The user should understand the core; the agent does the other modules; the user takes big decisions | Agreed, as three levels: executive, manager, execution | [direction §6](../direction.md#6-roles) |
| Flow: user brainstorms → agent judges → align on direction → agent executes | Adopted as the working loop | [README.md](README.md), CLAUDE.md |
| Configs and checkpoints are chaos (HF, local weights, library-managed) | One table per model of where the weights come from; one environment per servant | [models/README.md](../models/README.md) |
| SAM3: main spatial engine (auto-labelling, video tracking, concept prompts, panoptic by nature) | Agreed: the spatial teacher | [direction §5](../direction.md#5-model-roles) |
| RF-DETR for compute, YOLO for edge (tiers) | Agreed: they are students. YOLO first for the holy grail, RF-DETR as fallback | D-17, [models](../models/README.md) |
| Train Mask2Former from SAM3 panoptic output | Parked; YOLO semantic segmentation covers stuff classes | D-06 |
| Sapiens (should be Sapiens2): skeletons, body parts, actor features after cropping | Agreed: the person specialist, top-down on crops | [ideas/actors.md](../ideas/actors.md) |
| DINOv3 patch features (spatial); EmbeddingGemma for RAG | Agreed: curation and search | [ideas/search.md](../ideas/search.md), [holy_grail](../ideas/holy_grail.md) |
| BoxMOT is easy to use, a headache to tune; Ray/Optuna broke on WSL2; is ReID needed?; preparing raw frame detections clashed with v1 | Tuning belongs to deployment. Pseudo-ground-truth from SAM3 tracks. ReID is a variable to tune | [ideas/tracker_tuning.md](../ideas/tracker_tuning.md) |
| Actions: which frame exactly is ambiguous; different tasks need different fps; Qwen 4B ≈ 1 min, 8B ≈ 10 s of video; drawing boxes and asking which actor does what works surprisingly well; VLM temporal localisation fails | Window classification; accept stride precision; keep the marked-boxes trick | D-05, [ideas/vlm_window_actions.md](../ideas/vlm_window_actions.md) |
| Video tubelet models (VideoMAE2) retired; TempBridge on DINOv3 is untested for lack of data | Right call. VLM window labels become the missing eval data | [ideas/vlm_window_actions.md](../ideas/vlm_window_actions.md) |
| Actors are 99% humans; others deserve something | Generic tracks for any class; specialists only for people | [ideas/actors.md](../ideas/actors.md) |
| Don't write wrapper engines; read model cards and install directly (Qwen install was a WSL2 headache) | Agreed. Servants call vendor APIs; docs record only what the card doesn't say | D-01, [models/vlm.md](../models/vlm.md) |
| Holy grail: feature embeddings, statistics for a good spread, filter static frames, closed-loop iteration, verification on real data | Not foolish: established practice (coverage / coreset). Adopted as the first goal, with an experiment that proves it | D-04, [ideas/holy_grail.md](../ideas/holy_grail.md) |
| Rules: no pedantry, no sycophancy, less ambiguity; the user runs git and outside commands; 16 GB card | Carried over | [CLAUDE.md](../../../CLAUDE.md) |
