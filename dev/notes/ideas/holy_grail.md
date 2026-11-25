# Holy grail: choosing what to label from raw video, and proving it helped

> Status: **adopted as the first goal** (D-04, D-17). This file holds the idea; the plan is in [../milestones/README.md](../milestones/README.md).

## 1. The problem

Computer vision has no closed loop on real data. Raw footage is endless and mostly redundant, so labelling all of it is impossible, and choosing frames at random spends most of the budget on near-identical scenes. Nobody can tell from the data alone whether a training set is *good*. The holy grail is to automate the choice, and to measure the result instead of trusting it.

## 2. The loop

```mermaid
flowchart TB
    RAW["Raw video<br/>MEVA pool: 2 cameras × 1 h"] --> DEC["0 · Decode<br/>sample 1–2 fps"]
    DEC --> FIL["1 · Cheap filters (no model)<br/>static · blur · exposure"]
    FIL --> EMB["2 · Embed<br/>DINOv3"]
    EMB --> DED["3 · De-duplicate<br/>cosine cutoff"]
    DED --> SEL["4 · Select for coverage<br/>within a label budget N"]
    SEL --> LAB["5 · Label<br/>SAM3 'person' + quality gates"]
    LAB --> CHK["6 · Check<br/>person judges (D-02)"]
    CHK --> TRN["7 · Train student<br/>YOLO (fallback RF-DETR)"]
    TRN --> EVA["8 · Evaluate<br/>fixed, checked test set"]
    EVA --> MIN["9 · Mine<br/>student unsure / disagrees with SAM3"]
    MIN -->|"next round"| SEL
```

| Step | What | Notes |
|---|---|---|
| 1 Cheap filters | Frame difference or optical flow below a noise floor; variance of the Laplacian (blur); histogram clipped at either end (exposure) | From v1's sampling notes ([stream](../../rin_v1/docs/notes/prefilter_sampling/fiter_raw_stream_tier123.md), [static images](../../rin_v1/docs/notes/prefilter_sampling/static_images_sampling.md)); the stream note scores motion on a 320×180 copy to keep it cheap. CPU only. On MEVA's busy gym hour it removes little: motion ≥ 0.002 (below that the gym is empty) keeps 98%, and no frame is blurred or badly exposed (M2). Idle footage would lose far more |
| 2 Embed | DINOv3 CLS (maybe mean patch) per kept frame | On a fixed camera this may be the wrong signal (§4) |
| 3 De-duplicate | Drop frames too similar to one already kept | |
| 4 Select | Spend budget N for coverage of the embedding space | Methods in §3 |
| 5 Label | SAM3 concept prompt "person" gives boxes and masks | Gates: score floor; drop tiny or frame-filling masks |
| 6 Check | A person accepts or rejects frames and records | Needed above all for the test set |
| 7 Train | YOLO on the exported set (native format, D-12) | RF-DETR if YOLO hits a dead end |
| 8 Evaluate | mAP on the fixed test set | §5 |
| 9 Mine | Frames where the student is unsure or disagrees with SAM3 feed the next round | Makes it a loop, not a one-off |

## 3. Selection methods to compare

```mermaid
flowchart LR
    P["Candidate frames<br/>(after steps 1–3)"] --> R["Random in time<br/>baseline"]
    P --> F["Farthest-point (coreset)<br/>max coverage, loves outliers"]
    P --> K["Cluster-stratified<br/>k-means, equal picks per cluster"]
    P --> U["Uncertainty<br/>round 2+, needs a student"]
```

- **Random in time:** the baseline.
- **Farthest-point:** repeatedly pick the frame farthest from everything already picked. It maximises coverage but loves outliers: a glitch or a blurred frame is "far" from everything, which is why the step 1 filters come first. v1's notes start it from a "high-energy" frame (not defined there; most motion is one reading) rather than a random one. A fixed start makes all seeds pick the same list, so the seed would then only vary the student's training.
- **Cluster-stratified:** k-means, then an equal number of frames from each cluster, sampled *within* the clusters, because picking only centroids favours the most generic frames.
- **Uncertainty:** frames the current student scores at middling confidence, or where it disagrees with SAM3.

The user's instinct for a "good standard deviation" is what coverage measures; the experiment turns it into a number.

## 4. The main risk: fixed cameras

On a fixed CCTV camera the background never changes, so whole-frame embeddings of two different moments can be almost identical. The variety lives in the **foreground**: who is where, doing what. If whole-frame selection does no better than random, try foreground-aware features:

```mermaid
flowchart LR
    F["Frame"] --> A["Cheap detector → person crops<br/>→ DINOv3 per padded crop<br/>→ pool per frame"]
    F --> B["Patch tokens − per-camera background<br/>(median tokens over time)<br/>→ pool changed patches"]
    A --> E["Foreground embedding"]
    B --> E
```

Measured in M2 ([research](../../research/m2_filters_dedupe/README.md)): two random frames of one camera have a median whole-frame cosine of 0.95, so the background does dominate. One person crossing an empty gym moves the vector by 0.006.

Actor features, in order to try: (1) remove each camera's mean vector; (2) patch tokens minus the camera's mean patch grid, pooled where they differ (no detector, no class). Ruled out for selection: SAM3 (circular: it is the teacher being budgeted), Sapiens2 (built for pose, heavy). Stock YOLO boxes are possible but messy, and miss the small, occluded people most worth labelling.

This is the most likely place the idea fails at first, and the most interesting one to get right. A first negative result means "try §4", not "the idea failed".

## 5. The proof

**Experiment:** for label budgets N (e.g. 100, 250, 500, 1000 frames), train the same YOLO student with each selection method, over 3 seeds, and plot test mAP against N.

**Success (D-17):** curated selection reaches random's mAP with half the labels or fewer.

| Fixed part | Value |
|---|---|
| Class | person |
| Student | YOLO; one size fixed for the whole experiment (fallback RF-DETR) |
| Pool | MEVA school site, cameras G299 and G330, 2018-03-07, one continuous hour ([../data.md](../heavy_data.md)) |
| Test set | Same cameras, **another day** (2018-03-09). Its SAM3 labels are checked by a person before any experiment runs, and selection never touches it (D-13) |
| Val set | A few whole recordings held out from the pool day, SAM3 labels unchecked; only picks each student's checkpoint (D-31) |

## 6. Known risks

- **Pseudo-label noise.** SAM3 misses objects, and a missed person in a training frame teaches the student "background". Remedy: reject whole frames or spans (judgments). A signal for where to look that needs no person (the user's idea, from v1's [sampling note](../../rin_v1/docs/notes/prefilter_sampling/static_images_sampling.md), Stage 3): propagate SAM3's masks from frame t to t+1 and compare them with a fresh segmentation of t+1; a sharp IoU drop on a still camera flags an unstable label. Costs a second SAM3 pass per checked frame.
- **Small people.** MEVA cameras are wide and far away, so SAM3 may miss distant people unless frames are tiled. Measure on the test set before blaming selection.
- **Leakage.** Never split by frame (D-13).
- **Thresholds don't travel.** M2's numbers (motion ≥ 0.002, cosine 0.99) fit this gym, this model and this input size; another camera or model shifts them. What transfers is a rule set *relative to each stream*: a cut-off taken from the stream's own distribution (e.g. drop frames closer than a typical 0.5 s neighbour), vectors with the camera's mean removed, or a budget instead of a threshold. The fully general signal comes in round 2: the student's own uncertainty is defined by the model being trained, not by the footage. A VLM fits semantic filters ("frames with someone carrying a box", [search.md](search.md)), not near-duplicate removal.
- **Cost.** SAM3 time per frame limits the budgets we can afford. Measure it early.

## 7. Later

- Text-targeted mining with EmbeddingGemma 2 ("people carrying boxes"): [search.md](search.md).
- The trained student becomes the cheap screener in step 1, which closes the loop in deployment.
