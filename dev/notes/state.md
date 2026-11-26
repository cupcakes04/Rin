# State: where we are now

> The handoff between sessions. The agent rewrites this file at the end of every session ([CLAUDE.md, "How we iterate"](../../CLAUDE.md#how-we-iterate)). Current truth only.

**Updated:** 2026-11-26

## Phase

```mermaid
flowchart LR
    A["Direction<br/>✅ agreed"] --> D["M0 usage draft<br/>✅ agreed"] --> E["M1 core + ingest<br/>✅ closed"] --> F["M2 filters + embeddings<br/>✅ closed"] --> G["M3 teacher + checked test set<br/>▶ step 4: check 300 frames"]
    style G fill:#fde68a,color:#000
```

## Done
- **Direction and decisions:** D-01 … D-31 agreed ([decisions.md](decisions.md)). Design target: [usage.md](usage.md) on [data_model.md](data_model.md).
- **Data:** MEVA slice, sample media and v1 checkpoints in `workspace/` ([heavy_data.md](heavy_data.md)); Sapiens2 1B and DINOv3 ViT-L/16, ViT-H+/16 in the HF cache ([models/](models/README.md)).
- **M1 and M2 closed** (2026-11-25, user checked). What exists and its rules: [docs/core.md](../../docs/core.md); setup: [docs/install.md](../../docs/install.md). Store, ingest, sampling, the frame reader, `curate.score_frames` / `keep` / `center` / `dedupe`, the frame cache, and the first servant, `dinov3.embed`. 60 CPU tests.
- **M2's result** ([research/m2_filters_dedupe](../research/m2_filters_dedupe/README.md)): on the 2 fps pool, `motion >= 0.002` keeps 98%; `dedupe` at 0.99 leaves 4,453 candidates. Whole-frame vectors are dominated by the fixed background; centring by camera spreads them, and is for M4's selection.
- **Working rules** ([CLAUDE.md](../../CLAUDE.md)): the agent may run GPU work; debt is "to fix" (milestone file) or "parked" (with a trigger, [milestones/](milestones/README.md#parked-debt)); a running milestone is one file, `milestones/M<n>.md`; after a milestone the user may give [dev/audit.md](../audit.md) to an outside agent (another model); its report goes to `dev/audits/` and the agent triages it (process redesigned 2026-11-26 after the first audit).
- The inbox [brainstorms/brainstorm.md](brainstorms/brainstorm.md) is empty.

## Next
**M3 is running** ([milestones/M3.md](milestones/M3.md)).
- **Steps 1 and 2:** done.
- **Step 3, the blind pilot:** done ([research/m3_pilot](../research/m3_pilot/README.md)). 180 people on 18 frames, boxed in 22 minutes. SAM3 at score > 0.5 finds 93% of them, and 85% of its boxes are right.
- **Next, step 4:** check all 300 test frames, starting from SAM3's boxes, as proposed in D-33.

The first audit is triaged ([dev/audits/2026-11-26.md](../audits/2026-11-26.md#triage)); kept items wait for M4 and M5.

## Waiting on the user
- **D-33** ([decisions.md](decisions.md#proposed)): the checked test labels are one `human.annotate` run, started from SAM3's boxes above 0.5. A frame counts as complete when it is in that run. No judgments table for now.
- **The wrong-box sheet** ([false_at_0.5.jpg](../research/m3_pilot/false_at_0.5.jpg)): which of the 30 are real people.
- **Commit:** the viewer, `human`, the tests, `dev/scripts/fiftyone_mongod.sh`, `docs/`, the notes, `dev/research/m3_pilot/`.

## Vendor feature requests (to the user)
None open.

## Repo facts a new session needs
- Gitignored: `dev/rin_v1/`, `workspace/`, `.venv/`.
- To re-download MEVA: `bash dev/scripts/pull_meva.sh` (skips files already on disk).
- Core environment: `uv sync`; tests: `uv run pytest`. DINOv3 job environment: `uv sync --locked --project src/rin/servants/dinov3`.
- `workspace/store` holds a person's work from M3 on: never delete it to start over. Only its frame cache, `store/frames/`, is safe to delete. `workspace/store_m2/` is the M2 store, kept for the runs M2's research cites (its `frames/` is 11 GB).
