# Milestones

> Planned **one ahead** (D-19): the next milestone gets a detailed file when it starts; the rest stay one line each and are reshaped by what we learn. A finished milestone's file is folded into the docs or deleted.

```mermaid
flowchart LR
    M0["M0<br/>usage draft<br/>(teaching)"] --> M1["M1<br/>core + ingest"] --> M2["M2<br/>filters + embeddings"] --> M3["M3<br/>teacher + checked test set"] --> M4["M4<br/>selection + student"] --> M5["M5<br/>the proof"] --> M6["M6<br/>the loop"]
    style M0 fill:#fde68a,color:#000
```

| ID | Milestone | Done when | Status |
|---|---|---|---|
| M0 | **Usage draft** | The agent has written the notebook code the holy grail loop *should* read like, as a teaching document: each call explained, plus the data model it implies. The user has read it, asked questions, edited it and approved it. That settles the data model | next |
| M1 | Core + ingest | MEVA media in the store with frame indexes and pts; queries work; CPU tests pass | — |
| M2 | Cheap filters + embeddings | We know how many frames survive each filter, and the cost per frame | — |
| M3 | Teacher + checked test set | SAM3 labels the test day; the user checks them in FiftyOne (D-16) and the judgments are stored; SAM3's cost and misses are measured | — |
| M4 | Selection + student | Three selection methods; YOLO export; one student per budget | — |
| M5 | The proof | mAP against label budget per method, 3 seeds; an honest verdict (D-17) | — |
| M6 | The loop | Uncertainty mining for round 2 | — |

**Why the agent drafts M0:** "usage first" was the part of v1 that held up, and the user wants to learn by reading a worked draft rather than starting from a blank page. The user still decides: nothing in the core is final until the user approves the draft.
