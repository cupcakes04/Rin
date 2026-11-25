# Milestones

> Planned **one ahead** (D-19). The running milestone gets one file, `M<n>.md`: scope, open points, measurements, and its debt to fix. When it is done, what lasts is folded out of it (into `dev/notes/`, or `docs/` for user-facing docs), parked debt moves into the table below, and the file is deleted. The rest stay one line each and are reshaped by what we learn.

```mermaid
flowchart LR
    M3["M3<br/>teacher + checked test set"] --> M4["M4<br/>selection + student"] --> M5["M5<br/>the proof"] --> M6["M6<br/>the loop"]
    style M3 fill:#fde68a,color:#000
```

| ID | Milestone | Done when | Status |
|---|---|---|---|
| M3 | Teacher + checked test set | SAM3 labels the test day; the user checks them in FiftyOne (D-16) and the judgments are stored; SAM3's cost and misses are measured | next |
| M4 | Selection + student | Three selection methods; YOLO export; one student per budget | — |
| M5 | The proof | mAP against label budget per method, 3 seeds; an honest verdict (D-17) | — |
| M6 | The loop | Uncertainty mining for round 2 | — |

What is built so far is described in [docs/](../../../docs/core.md). The API every milestone builds towards is [usage.md](../usage.md), on the data model in [data_model.md](../data_model.md). Both stay in `dev/notes/` as the design target, because they describe parts not built yet. Root `docs/` only describes what exists: each milestone adds what it built there (install, the calls and tables that now work).

## Parked debt

Shortcuts that are fine to leave until their trigger happens. Each milestone start reads this table; a trigger that has fired turns its row into work for that milestone. Debt that the work will hit anyway is not parked: it goes in the milestone file as "to fix" and is fixed before the milestone closes.

| Shortcut | Trigger: do it when | Cost of doing it |
|---|---|---|
| Every run records Rin's git commit, so Rin must run from a git checkout | Rin is installed as a package somewhere | A version string as the commit's stand-in |
| Ingest decodes every frame to count it (about 4.3 s per 5-minute 1080p clip) | Ingest of a new dataset takes long enough to hurt | Reading packets is faster but can miscount; needs a check against decoding |
| Progress is plain `print` lines | Rin runs as a background job, not in a notebook | Switch to `logging` |
| Two `ingest` calls at once can both pass the "already ingested" check | Two processes write the store at the same time | A lock file around ingest |
| A truncated MKV isn't caught (MKV states no frame count) | MKV footage arrives | Compare against the container's duration instead |
| Launching a job (`uv run` in the servant's folder, streaming its output, checking the exit code) lives inside `dinov3/__init__.py` | A second servant is built (SAM3, M3) | Move it beside `frame_cache.py` as shared plumbing; each caller keeps its own arguments and outputs (D-01) |
| The frame cache stores full-resolution JPEGs at quality 95: about 0.78 MB per 1080p frame (11 GB for the 2 fps pool) | Disk runs short, or footage grows to many hours | Lower quality, or cache at the largest size a servant reads |
