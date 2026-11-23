# Actors: people first, without hard-coding people

> Status: **raw.**

- About 99% of actors are people, but the data model doesn't assume it: a track can belong to any class, at no cost.
- Person-only models (Sapiens2: 308 keypoints, body-part segmentation, actor features) run on person crops. They are top-down by design, so they take boxes from SAM3 or a detector as input.
- Actor features: Sapiens2 per person crop (identity, body), DINOv3 pooled per box (appearance and context). Which one serves re-identification and action heads better is an experiment, not a decision.
- Non-human actors (forklifts, robots, animals) get generic tracks and DINOv3 features. A specialist only if a real use case needs one.
