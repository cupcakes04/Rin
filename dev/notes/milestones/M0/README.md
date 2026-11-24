# M0 · Usage draft

> **In review.** Done when the user has read, questioned, edited and approved both drafts and answered D-22 … D-30 ([milestones](../README.md)).

| File | What |
|---|---|
| [usage.md](usage.md) | The holy grail loop as a notebook, steps 0–9; each cell explained (what / why / what breaks without it) |
| [data_model.md](data_model.md) | The tables that notebook implies, who writes and reads each, the layout on disk |

## Settled in review

- **Form:** Markdown with code cells, not `.ipynb`. **Scope:** the whole loop; parts C3 and D are sketches.
- **Call style:** one function per servant with its own arguments (`sam3.concept(m, frames, ...)`), never a generic `m.serve(...)`.
- **Ingest knows no dataset:** it records what the file says; camera and start time come from an optional `describe` function the user writes in the notebook (usage A1).
- **Missed people, cheapest fix first:** accept SAM3's low-score box → click and let SAM3 draw → draw by hand (usage B3, D-27).
- **Embeddings:** DINOv3 for selection; EmbeddingGemma 2 only if DINOv3 whole-frame doesn't beat random (usage A4).
- **Crowds:** about 30 people per frame is fine for per-frame SAM3 detection; tracking limits matter later ([models, gotchas](../../models/README.md#known-gotchas)).
- **Installation** gets real docs in `docs/install.md` in M1, once there is an environment to install ([milestones](../README.md)).

## Open

- **D-22 … D-30** ([decisions](../../decisions.md#proposed)): yes, no, or rewrite.
- **Where clicks and drawn boxes are collected** (usage B3). The user is unsure; the agent brings one recommendation at the start of M3, after checking what FiftyOne can do.

## When M0 is done

`data_model.md` moves to `dev/notes/data_model.md` as the core's current truth. `usage.md` turns into real notebooks as M1–M6 build each part, and its explanations into `docs/`. This folder is then deleted.

Debt the draft takes on: [usage.md §Debt](usage.md#debt-this-draft-takes-on).
