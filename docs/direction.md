# Rin direction

> Living document. Changes only after the user agrees. Rewrite in place.

## 1. Intent

**Raw video in → curated, labelled, checked 2D data out → small models that run on cheap hardware.**

```mermaid
flowchart LR
    RAW["Raw video<br/>(hours, mostly redundant)"] --> SEL["Select<br/>what's worth labelling"]
    SEL --> LAB["Label<br/>teacher models"]
    LAB --> CHK["Check<br/>a person judges"]
    CHK --> TRN["Train<br/>student models"]
    TRN --> EVA["Evaluate<br/>fixed test set"]
    EVA -->|"where is the student weak?"| SEL
    TRN --> EDGE["Deploy<br/>edge / GPU"]
```

The hard, unsolved part is **Select**: choosing *which* frames of an endless stream to label, and proving the choice helped. That is the **holy grail** and the first goal ([ideas/holy_grail.md](ideas/holy_grail.md)).

Rin is a **proof of concept that must still be built properly**: one user, one machine, one 16 GB GPU. It grows into a toolkit later. Quality comes from a sound data model, not from infrastructure.

## 2. The core principle

**Standardise the data, not the models.**

The models differ in every way: inputs, prompts, outputs, environments, checkpoints, memory limits. Forcing them behind one interface made v1 bloated and buggy ([lessons/rin_v1.md](lessons/rin_v1.md)). The data does not vary: media, frames, boxes, masks, keypoints, tracks, action spans, embeddings, and where each came from.

```mermaid
flowchart TB
    subgraph CORE["Core: small, owned by the user"]
      DM["Data model<br/>media · frames · instances · tracks · actions · embeddings"]
      ST[("Store<br/>runs + judgments")]
      PV["Provenance<br/>which model, checkpoint, config, input"]
    end
    S1["SAM3 servant"] --> ST
    S2["DINOv3 servant"] --> ST
    S3["YOLO servant"] --> ST
    S4["… one per model,<br/>no shared base class"] --> ST
    V1[["vendor API"]] -.-> S1
    V2[["vendor API"]] -.-> S2
    V3[["vendor API"]] -.-> S3
```

## 3. Naming: Master and Servants

| Name | What it is |
|---|---|
| **Master** | Owns the store. Holds media, runs and judgments, answers "what is true now" as a query, and sends work to servants. The object the user holds in a notebook |
| **Servant** | One per model (e.g. the SAM3 servant). Runs its model as a batch job in that model's own environment, and returns a **run** |
| **Run** | One servant call: its records, plus the model, checkpoint, config and inputs used. Never edited after it is written |
| **Judgment** | A person's (or a rule's) accept or reject on a record, a frame or a time span. Stored apart from runs |

```mermaid
sequenceDiagram
    participant U as User (notebook)
    participant M as Master
    participant S as Servant (own env, batch job)
    participant D as Store (disk)
    U->>M: serve SAM3 on these frames, prompt "person"
    M->>S: start job (inputs + config)
    S->>D: write run (records + provenance)
    S-->>M: job finished, run id
    U->>M: judge: reject frame 812 (missed people)
    M->>D: write judgment
    U->>M: dataset = accepted records of run X
    M->>D: query runs × judgments
    M-->>U: training set
```

The names come from v1; the meaning is new. In v1 the Master was a mutable catalog with branches, and the Servant pushed requests to long-running HTTP services (D-02, D-07).

## 4. Scope and order

**In:** everything 2D in video and images. **Out:** 3D, depth, sensor fusion; a later project can consume Rin's outputs as long as each record carries its camera and time.

```mermaid
flowchart LR
    G["1 · Holy grail<br/>spatial: person detection"] --> A["2 · Actions<br/>VLM windows → DINOv3 head"]
    G --> T["3 · Tracking<br/>BoxMOT tuned for edge"]
    A --> AN["4 · Analysis + search"]
    T --> AN
    AN --> R["5 · Robotics hand-off<br/>timestamped 2D per camera"]
```

Ideas for each: [vlm_window_actions](ideas/vlm_window_actions.md) · [tracker_tuning](ideas/tracker_tuning.md) · [search](ideas/search.md) · [actors](ideas/actors.md).

## 5. Model roles

```mermaid
flowchart LR
    subgraph Teachers
      SAM3["SAM3 / 3.1<br/>boxes · masks · tracks · concepts"]
      SAP["Sapiens2<br/>pose · body parts (person crops)"]
      VLM["Qwen3-VL · Gemma 4<br/>actions (what, not when)"]
    end
    subgraph Features
      DINO["DINOv3<br/>patch + CLS"]
      EMB["EmbeddingGemma 2<br/>text ↔ image/video"]
    end
    subgraph Students
      YOLO["YOLO<br/>edge · first choice"]
      RFD["RF-DETR<br/>GPU tier · fallback"]
      HEAD["Action head<br/>on DINOv3 features"]
    end
    TRK["BoxMOT<br/>tracking on edge"]
    DINO -->|curation| SAM3
    EMB -->|text mining| SAM3
    SAM3 --> YOLO & RFD
    SAP --> YOLO
    VLM --> HEAD
    DINO --> HEAD
    YOLO --> TRK
```

Mask2Former is parked (D-06). Details per model: [models/README.md](models/README.md).

## 6. Roles

| Level | Who | Owns |
|---|---|---|
| Executive | User | Intent, direction, what counts as success |
| Manager | User | The data model and the API. The agent drafts the usage code as a teaching document; the user reads, questions, edits and approves it |
| Execution | Agent | Servants, store, jobs, environments, milestones, tests. Large decisions go to the user with a recommendation |

## 7. How we work

See [CLAUDE.md, "How we iterate"](../CLAUDE.md#how-we-iterate): the user brainstorms, the agent ingests, the user answers decisions, then milestones run, with `state.md` as the handoff between sessions.
