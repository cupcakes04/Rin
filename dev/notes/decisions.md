# Decisions

> Current truth only: rewrite an entry when it changes, delete it when it no longer holds. IDs stay stable once code cites them; gaps are fine.
> **Agreed** = the user said yes. **Proposed** = the agent recommends it and is waiting for a yes or no. **Open** = needs the user's call.

## Agreed

**D-01 · Standardise the data, not the models.** The core is the data model, the store and provenance. Servants are thin, separate adapters with no shared base class and no common request type. A model's special abilities stay in its own servant's arguments. *Why:* v1's three-pillar contracts were about 60% of the SDK and the source of most bugs ([lessons/rin_v1.md](lessons/rin_v1.md)).

**D-02 · Runs are immutable; judgments are separate; "what is true now" is a query.** Every servant call writes a new run (records plus model, checkpoint, config, inputs), and nothing is edited in place. Accept and reject go into a judgments table, per record, frame or time span. The training dataset is a query over runs and judgments. *Why:* this removes v1's branches, commit, fork, id remapping and shared nested objects, which together caused v1 CR-12, CR-32 and Studio debts A1–A5 and A8. Provenance comes for free.

```mermaid
flowchart LR
    R1["Run 1 · SAM3<br/>(immutable)"] --> Q{{"query"}}
    R2["Run 2 · SAM3, new prompt<br/>(immutable)"] --> Q
    J["Judgments<br/>accept / reject<br/>record · frame · span"] --> Q
    Q --> DS["Dataset now"]
```

**D-03 · Naming: Master and Servants**, as defined in [direction §3](direction.md#3-naming-master-and-servants).

**D-04 · The holy grail is the first goal.** Everything is built around it; actions, tracking and analysis come after ([direction §4](direction.md#4-scope-and-order)).

**D-05 · Actions are window classification.** We accept that action timing is only as precise as the window stride, and do not attempt VLM temporal localisation ([ideas/vlm_window_actions.md](ideas/vlm_window_actions.md)).

**D-06 · Mask2Former is parked.** Stuff classes (sky, road) come from YOLO semantic segmentation. Revisit for an in-house DINOv3-backbone model.

**D-15 · The toolkit is a Python library first.** Master and Servants are driven from notebooks and a small command line. No HTTP API or web app of our own until a concrete need forces one: the viewer can't do something review needs, or another machine or program must call Rin.

**D-16 · FiftyOne serves as a disposable viewer.** It reads a throwaway export (one script) and is never the data model.

**D-17 · Holy grail setup:**
- class "person" first;
- student YOLO, with RF-DETR as the fallback if YOLO hits a dead end;
- selection embeds frames with DINOv3; EmbeddingGemma 2 is tried only if whole-frame DINOv3 doesn't beat random ([usage A4](usage.md#a4--embed));
- success means curated selection reaches random selection's test mAP with half the labels or fewer.

**D-18 · Servants are rewritten, not ported from v1.** v1's engines are shaped around its pillar contracts (`setup`, `categories`, `ground_*`, HTTP apps), and the user's vendor engines (`sam3_engine`, `gemma_emb_engine`) already are the clean API. v1 stays a source of *facts* (gotchas, working settings, label lists). Its pure helpers are salvaged one at a time, with the user's OK ([lessons §Salvage](lessons/rin_v1.md#salvage-code-worth-keeping)).

**D-19 · Milestones are planned one ahead.** Only the next milestone gets a detailed file; later ones stay one line each and are reshaped by what we learn. *Why:* v1's plans ballooned before the first loop ran.

**D-07 · Servants run as batch jobs, not daemons.** A job runs in its model's environment, reads inputs from disk, writes its run to disk and exits, which frees the VRAM. No ports, no PM2, no VRAM governor. A long-lived process only for an interactive need (e.g. SAM3 clicks, VLM chat), and only when that need arrives. *Why:* v1's stale SAM3 prompts (CR-25) and governor races came from long-lived services.

**D-08 · Records are stored as Parquet and queried with DuckDB; features go in plain array files.** *Why:* v1's JSON catalog took 2.35 s per commit at 250k records, and SQL replaces its five hand-written indexes.

**D-09 · The frame index is the join key; pts is stored beside it.** Frames are decoded with the container's real per-frame timestamps (v1 ADR 5; the node10 cameras run at 14.25 and 14.33 fps). *Why:* v1 matched float timestamps within a tolerance and silently lost frames (CR-30).

**D-10 · Geometry is normalised to [0, 1] in the store.** Pixels only at the edges (servant input and output, exporters, viewers). Media records carry width and height. *From v1 ADR 4.*

**D-11 · Keypoints are matched by name, never by position.** *From v1 ADR 16:* Goliath is not ordered like COCO (its wrists are points 62 and 41).

**D-12 · Exporters write each training framework's own format** (YOLO, COCO, RF-DETR, MOT). No generic dataloader. *From v1 ADR 2.*

**D-13 · Splits are made by recording, never by frame.** The holy grail's test set comes from a different day than its pool, and a person checks its labels. *Why:* frames of one video are near-duplicates, so splitting by frame leaks. And a test set labelled by SAM3 alone measures agreement with SAM3, not accuracy.

**D-14 · Feature storage is selective.** By default, store CLS vectors and per-box pooled vectors. Compute patch grids on demand, or cache them only for selected frames. *Why:* patch grids take megabytes per frame, so hours of video would take tens of GB.

**D-20 · `uv` manages every environment.** The core and each servant get their own `uv` venv; the user's engines are installed with `pip install -e vendors/<engine>`. System FFmpeg replaces what conda provided. *Why:* one fast tool with lockfiles, instead of conda plus pip.

**D-21 · One gitignored `workspace/` holds everything heavy and local:** `media/`, `checkpoints/` and, from M1, `store/`. The root stays small: `CLAUDE.md`, `dev/` (notes, research, scripts, tests, and the gitignored `rin_v1/`), `docs/` (the repo's real docs, from M1), `src/`, `vendors/`, `workspace/`, plus `pyproject.toml`. *Why:* the user's v1 habit; it keeps the root tidy, and keeps notes about building Rin apart from docs for someone using it.

**D-22 · Everything that produces data is a run, and every step has one shape:** `module.verb(m, inputs, **own_args) -> Run`. That includes ingest, sampling, filters, selection, set operations, exports, training, evaluation and a person's drawn boxes (`human.annotate`). The Master only holds the store, judgments and queries. A servant is a caller in the core environment (real signature, checks arguments, no model imports) plus a job in the model's environment. *Why:* one provenance mechanism for everything, and a typo fails before the GPU starts.

**D-23 · One folder per run, write once; no counters.** A job writes into a temporary folder and renames it into `store/runs/` as its last act, so a crashed job leaves nothing. Tables are DuckDB views over the runs' Parquet files; no file is ever rewritten. `run_id` is time + step + 4 random characters; records are keyed `(run_id, index)`. Judgments go in one new file per judging session. *Why:* v1's id counters, remapping and whole-catalog rewrites (CR-12, 2.35 s per commit) can't happen.

**D-24 · An image is a media with one frame.** One media table, one frames table, one instances table. *Why:* halves v1's record types with no loss.

**D-25 · Classes are plain strings; boxes are four [0, 1] columns.** No category table or alignment dicts: a servant maps its prompts to class names in its own arguments. Boxes are `x0, y0, x1, y1`; masks are COCO RLE; keypoints a list of `{name, x, y, score}`. *Why:* the simplest thing that queries well in SQL.

**D-26 · Measure once, decide cheaply.** Expensive steps store everything they produce above a low floor (SAM3's `score_floor`, every filter score). Thresholds and quality gates are applied later, in a cheap step or the dataset query, and saved in that run's config. *Why:* changing a gate never re-runs the teacher or re-decodes video.

**D-27 · What a judgment means.** Targets: an instance, a run's labels on a frame, or a span. A frame *accept* means every object of the class on that frame is labelled correctly. An instance *accept* keeps a box even below the score gate. Missing objects are fixed cheapest first: accept SAM3's low-score box; else a person clicks and SAM3 draws (`sam3.click`); else a person draws (`human.annotate`). Added boxes are runs with the person in `by`. The latest judgment on a target wins; nothing is deleted. *Why:* a missed person isn't a box anyone can reject, and a test set needs complete frames.

**D-28 · Selections are ranked; budgets are prefixes.** Each selection writes up to `n_max` frames in order; budget N takes the first N. SAM3 labels the union of all selections once. *Why:* nested budgets give smoother curves and cut SAM3 time, the most expensive step.

**D-29 · Frame images are decoded once into a cache in the store** (`store/frames/<media_id>/<frame_idx>.jpg`), the first time a servant needs them. Derived data: safe to delete. *Why:* no servant environment needs its own video decoding. *Cost:* JPEG is a second lossy compression; identical for every model, so comparisons stay fair.

**D-30 · The test set is a random sample of the test day, never curated, and a person checks every frame.** Empty frames stay in. *Why:* a curated test set rewards the method that curated it; empty frames are where false positives show up.

## Proposed

None right now.

## Open

**Where a person's clicks and drawn boxes are collected** ([usage B3](usage.md#b3--a-person-checks-it)). FiftyOne tags frames and boxes; whether it can also collect clicks for `sam3.click` and hand-drawn boxes is unchecked. The agent brings one recommendation at the start of M3.
