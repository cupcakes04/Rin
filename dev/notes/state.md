# State: where we are now

> The handoff between sessions. The agent rewrites this file at the end of every session ([CLAUDE.md, "How we iterate"](../../CLAUDE.md#how-we-iterate)). Current truth only.

**Updated:** 2026-11-25

## Phase

```mermaid
flowchart LR
    A["Direction<br/>✅ agreed"] --> D["M0 usage draft<br/>✅ agreed"] --> E["M1 core + ingest<br/>✅ closed"] --> F["M2 filters + embeddings<br/>📝 scoping"]
    style F fill:#fde68a,color:#000
```

## Done
- **Direction and decisions:** D-01 … D-31 agreed ([decisions.md](decisions.md)). The design target is [usage.md](usage.md) (the notebook the code must read like) on [data_model.md](data_model.md) (the tables).
- **Data:** MEVA slice, sample media and v1 checkpoints in `workspace/` ([heavy_data.md](heavy_data.md)). Sapiens2 1B and DINOv3 ViT-L/16 and ViT-H+/16 in the HF cache ([models/](models/README.md)).
- **M1 closed** (2026-11-25, user checked): store, ingest and sampling, 36 CPU tests. What exists and its rules: [docs/core.md](../../docs/core.md); setup: [docs/install.md](../../docs/install.md). The MEVA pool and test day are in `workspace/store`. Its debt moved to [milestones/M2/](milestones/M2/README.md#debt).
- **External audit** (2026-11-25), each claim checked against the code and v1. It gave M1's truncation and early-check fixes, D-31 (val split), the M3 open question on fixed frames, and v1 facts now in [lessons/rin_v1.md](lessons/rin_v1.md), [models/](models/README.md) and [ideas/holy_grail.md](ideas/holy_grail.md).
- The inbox [brainstorms/brainstorm.md](brainstorms/brainstorm.md) is empty.

## Next
1. **The user answers M2's three open points** ([milestones/M2/](milestones/M2/README.md#open-points-the-users-call)): who writes a servant's run, DINOv3 input size, where a servant's environment lives.
2. Then the agent builds M2's CPU parts (frame reader, `score_frames`, `keep`, frame cache, `dedupe`, the DINOv3 caller), and hands over the DINOv3 install and embed as numbered GPU steps.

## Waiting on the user
- M2's open points (above).
- Commit: `pyproject.toml`, `uv.lock`, `src/`, `docs/`, `dev/tests/`, `dev/notes/` changes (including the rename `data.md` → `heavy_data.md` and the deleted `milestones/M1/`), `dev/scripts/pull_meva.sh`, `CLAUDE.md`, `.gitignore`.

## Vendor feature requests (to the user)
None open.

## Repo facts a new session needs
- Gitignored: `dev/rin_v1/`, `workspace/`, `.venv/`.
- To re-download MEVA: `bash dev/scripts/pull_meva.sh` (skips files already on disk).
- Core environment: `uv sync`; tests: `uv run pytest`. `workspace/store` is derived data: delete it and re-ingest (about 130 s).
