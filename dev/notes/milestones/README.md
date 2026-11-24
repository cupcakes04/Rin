# Milestones

> Planned **one ahead** (D-19): the running milestone gets a folder, `M<n>/`, holding its README (scope, open points, debt) and its working drafts. The rest stay one line each and are reshaped by what we learn. When a milestone is done, what lasts moves out of its folder (into `dev/notes/`, or `docs/` for user-facing docs) and the folder is deleted.

```mermaid
flowchart LR
    M1["M1<br/>core + ingest"] --> M2["M2<br/>filters + embeddings"] --> M3["M3<br/>teacher + checked test set"] --> M4["M4<br/>selection + student"] --> M5["M5<br/>the proof"] --> M6["M6<br/>the loop"]
    style M1 fill:#fde68a,color:#000
```

| ID | Milestone | Done when | Status |
|---|---|---|---|
| M1 | Core + ingest | MEVA media in the store with frame indexes and pts; queries work; CPU tests pass; `docs/install.md` sets up the core environment from scratch | next |
| M2 | Cheap filters + embeddings | We know how many frames survive each filter, and the cost per frame | — |
| M3 | Teacher + checked test set | SAM3 labels the test day; the user checks them in FiftyOne (D-16) and the judgments are stored; SAM3's cost and misses are measured | — |
| M4 | Selection + student | Three selection methods; YOLO export; one student per budget | — |
| M5 | The proof | mAP against label budget per method, 3 seeds; an honest verdict (D-17) | — |
| M6 | The loop | Uncertainty mining for round 2 | — |

The API every milestone builds towards is [usage.md](../usage.md), on the data model in [data_model.md](../data_model.md).
