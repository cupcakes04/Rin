# State: where we are now

> The handoff between sessions. The agent rewrites this file at the end of every session ([CLAUDE.md, "How we iterate"](../../CLAUDE.md#how-we-iterate)). Current truth only.

**Updated:** 2026-11-27

## Phase

```mermaid
flowchart LR
    A["Direction<br/>✅ agreed"] --> D["M0 usage draft<br/>✅ agreed"] --> E["M1 core + ingest<br/>✅ closed"] --> F["M2 filters + embeddings<br/>✅ closed"] --> G["M3 teacher + checked test set<br/>✅ closed"] --> H["M4 selection + student<br/>▶ step 3 of 5"]
    style H fill:#fde68a,color:#000
```

## Done
- **Direction and decisions:** D-01 … D-39 agreed ([decisions.md](decisions.md)); nothing Proposed or Open. Design target: [usage.md](usage.md) on [data_model.md](data_model.md).
- **M1, M2, M3 closed.** What exists and its rules: [docs/core.md](../../docs/core.md); setup: [docs/install.md](../../docs/install.md). 153 CPU tests.
- **The test set** (M3): run `20261127-094340-human.annotate-64f9`, 300 checked frames from five days ([m3_checked_test_set](../research/m3_checked_test_set/README.md)).
- **M4** ([M4.md](milestones/M4.md), started 2026-11-27; the user's calls: the gym stays for M4 and M5, SAM3 labels every candidate (D-37), COCO scoring moves into M4):
  - **Step 1 passed:** pool chain rebuilt in `workspace/store` (4,453 candidates, the same frames as M2's); val = 03-07 11:30–11:40 (D-31), 308 candidates; DINOv3 pinned and offline; streams by camera, day and split. `store_m2` deleted.
  - **Step 2 passed:** SAM3 on all 4,761 candidates (run `5477`, 1.36 s/frame); training labels per D-39 (score ≥ 0.5, no size gate, same-spot boxes kept: in the pool they are seated people).
  - **Step 3 built, waiting on the user's check:** `curate.select` (D-38), nine selections (3 methods × 3 seeds); measurements in M4.md: kcenter halves the worst-case distance and picks outliers, kmeans_stratified is close to random.

## Next
1. **The user checks step 3** (below), then **step 4**: `m.dataset`, `export.yolo`, the `yolo.train` servant, one training timed. Read v1's `src/rin/data/export/yolo/writer.py` first ([lessons](lessons/rin_v1.md)).
2. Step 5: `export.coco`, `coco.evaluate`, one student per budget.

## Waiting on the user
- **Step 3 check:** look at the first-50 sheets in [dev/research/m4_selection/](../research/m4_selection/) and the table in [M4.md](milestones/M4.md#measurements); pass if kcenter looks more varied than random.
- **Commit:** the M3 follow-up work (from the last session) and today's M4 work: notes (M4.md, decisions D-31/D-37/D-38/D-39, milestones, usage, heavy_data, ideas/review_by_exception, state), code (`curate/streams.py`, `curate/select.py`, dedupe/center grouping, DINOv3 revision, `human.suspects` chunking), tests, `pull_meva.sh` (val clips), research folders `m4_training_labels/`, `m4_selection/`.

## Vendor feature requests (to the user)
None open.

## Repo facts a new session needs
- Gitignored: `dev/rin_v1/`, `workspace/`, `.venv/`.
- To re-download MEVA (pool, val, test): `bash dev/scripts/pull_meva.sh` (skips files already on disk).
- Core environment: `uv sync`; tests: `uv run pytest`. DINOv3 job environment: `uv sync --locked --project src/rin/servants/dinov3`; its model commit is `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- `workspace/store` holds a person's work (the test set, the pilot): never delete it to start over; back it up with `dev/scripts/backup_human_runs.sh`. Only its frame cache, `store/frames/`, is safe to delete.
