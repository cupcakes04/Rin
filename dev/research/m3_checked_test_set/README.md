# The checked test set: what a person changed, and how reliable it is

**Question.** Starting from SAM3's boxes, with suspect frames flagged by code (D-33), what does a person change to make the 300 test frames true, and how far can the result be trusted?

**Setup** (M3 step 5, 2026-11-26). The 300 test frames of D-32 (`20261126-130121-curate.random-0187`), shown in FiftyOne view `check` with SAM3's 4,897 detector boxes above 0.5 (`20261126-205940-sam3.concept-11ee`, D-36) to edit, its 0.3–0.5 boxes as read-only hints, and 152 frames tagged `flagged` by `human.suspects` (407 boxes: 395 on the same spot on at least half of a camera's frames that day, 12 near-duplicates). Classes `person` and `crowd` (D-34). Filed as **`20261126-231507-human.annotate-9c6c`: the test set.** An earlier check of the same frames, started from the boxes around SAM3's masks with no flags, kept 99.4% of SAM3's boxes, bags included, and was discarded (audit [2026-11-26-m3](../../audits/2026-11-26-m3.md) §1.1).

**Time:** about 1 h 45 min from the first edit to the first save (FiftyOne's edit times; breaks included), then a short fix pass.

## Results

**The test set:** 300 frames, 3,464 people, 47 `crowd` boxes on 37 frames (1,193 of SAM3's boxes inside them dropped), 110 empty frames.

**What the person changed**, by how many boxes SAM3 put on the frame (outside crowd boxes):

| SAM3 boxes on the frame | frames | people | frames changed | SAM3 boxes deleted or moved | boxes drawn or moved | frames with a crowd box |
|---|---|---|---|---|---|---|
| none | 54 | 0 | 0 | 0 | 0 | 0 |
| 1–5 | 90 | 80 | 71 | 161 | 0 | 0 |
| 6–20 | 71 | 735 | 23 | 35 | 0 | 1 |
| more than 20 | 85 | 2,649 | 53 | 48 | 4 | 36 |

**SAM3 above 0.5 against the test set,** outside crowd regions, matched at IoU 0.5: 3,460 of its 3,704 boxes are people (precision 0.934). Its wrong boxes sit mostly on sparse frames: bags, coats and chairs on the bleachers, on the same spot across a whole window (03-13 and 03-15 G330). Recall is 0.999, but that number can't be trusted: a check that starts from SAM3's boxes rarely adds anyone (4 people drawn on 300 frames). The blind pilot is the honest measure of misses: SAM3 above 0.5 found 94% of the people a person boxed blind ([m3_pilot](../m3_pilot/README.md)).

**The flags found the errors:** 139 of the 152 flagged frames needed a change. Of the 116 frames with 20 SAM3 boxes or fewer and no flag, 2 needed one.

**Checked against the blind pilot** (the same person boxing 18 of these frames without SAM3, earlier): of its 180 boxes, 147 match a test-set person (IoU 0.3), 28 lie inside crowd boxes, and 5 don't match. Of those 5, one is a pilot box around two people (the test set has them apart), two are the 03-15 bag the pilot took for a person, and **two are people the test set misses** (G330 03-12 frame 7995: one seated at the front of the bleachers, one cut off at the left edge).

**Known errors left** (agent's look at the remaining flags and the pilot comparison): the 2 missed people above, a duplicate box on one head (G299 03-12 frame 8055, IoU 0.97) and a probable one (G299 03-12 frame 5010). About 4 boxes in 3,464.

## Verdict

- **The test set is usable:** every frame checked, its known errors about 0.1% of the people. Misses that neither SAM3 nor the check caught can't be counted from this data; the blind pilot found 2 such in 18 frames.
- **SAM3's error on this footage is objects, not people:** on the 144 frames with 0–5 SAM3 boxes, the person deleted 161 boxes and drew none. Unchecked SAM3 labels will teach a student that bags on bleachers are people; M4's training labels are unchecked (usage C2).
- **Flags make checking work:** with them, the errors the first check passed were caught. A rule for [review by exception](../../notes/ideas/review_by_exception.md) follows (trust frames with 20 boxes or fewer and no flag); its numbers come from the frames the flag thresholds were picked on, so they are optimistic.
- **Crowds:** a third of SAM3's boxes (1,193 of 4,897) fell in crowd regions, so the test set mostly scores people a person can see clearly. M5 scores with crowd regions ignored (D-34).
