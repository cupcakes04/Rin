# The checked test set: how much did a person change, and where?

**Question.** Starting from SAM3's boxes above 0.5 (D-33), how much does a person change to make all 300 test frames complete, how long does it take, and where are the changes? How reliable is the result where the person had to guess?

**Setup** (M3 step 4, 2026-11-26). The 300 test frames of D-32 (`20261126-130121-curate.random-0187`), shown in FiftyOne with SAM3's 4,897 boxes above 0.5 (`20261126-130121-sam3.concept-7efc`) to edit and its 0.3–0.5 boxes as a read-only hint layer. The user reviewed every frame; a second pass went over the 35 boxes wider than 1.5 × their height or taller than 400 px, tagged `wide?`. Filed as **`20261126-164727-human.annotate-71ed`: 4,884 boxes on 300 frames, the checked test set.**

## Results

**Time:** 14 minutes for the first pass over 300 frames, 10 for the `wide?` pass: 24 minutes, about 5 s per frame. The blind pilot took 22 minutes for 18 frames drawn from scratch ([m3_pilot](../m3_pilot/README.md)). The user: simple frames take seconds; crowded ones take the time, and some people there are a guess (a scarf on a chair or a person?).

**Edits:** 272 of 300 frames needed none. 4,854 of SAM3's 4,897 boxes stayed as they were (99.1%); 43 deleted, 30 drawn or moved.

| People on the frame | Frames | People | Frames edited | Deleted | Drawn or moved |
|---|---|---|---|---|---|
| none | 55 | 0 | 1 | 4 | 0 |
| 1–5 | 90 | 241 | 1 | 4 | 0 |
| 6–20 | 70 | 764 | 4 | 2 | 4 |
| more than 20 | 85 | 3,879 | 22 | 33 | 26 |

By day: 03-09 (11 of 57 frames edited), 03-12 (9 of 54), 03-11 (6 of 57), 03-13 (2 of 67), 03-15 (0 of 65): the edits follow the crowds.

**SAM3's one systematic error: group boxes.** When a mask runs across several people (a row of seated people, two players side by side) the box covers the group. They are what the user first took for "stretched" boxes, and deleting one uncovered people SAM3 had not boxed alone. Of the 35 boxes wider than 1.5 × their height (or taller than 400 px), the user deleted 10 as groups and kept 25 as real people (seated sideways, lying down): a box much wider than tall is worth a look, not a verdict.

**How reliable are the crowds?** On the 18 pilot frames the same person labelled twice: blind (pilot) and checked from SAM3. Matched at IoU 0.3, 169 people are in both, 11 only in the blind pass, 29 only in the checked one; nearly all differences are in crowds. Two careful passes disagree on about a fifth of the people in a dense crowd; in sparse frames they agree.

## Verdict

- **The checked test set exists:** 300 frames from five days, 4,884 people, 55 empty frames, every frame looked at by a person.
- **SAM3 plus a quick check is close to perfect away from crowds:** 6 of 215 frames with 20 people or fewer needed any edit. This is the first measured rule for [review by exception](../../notes/ideas/review_by_exception.md): trusting frames with 20 boxes or fewer would have spared 215 frames, 6 of them (1 in 36) with an error.
- **Crowds are noisy for anyone:** about a fifth of crowd people are disputed between two passes. M5 scores students on this set, so a difference between selection methods that shows only in crowded frames is suspect; parked debt with that trigger ([milestones](../../notes/milestones/README.md#parked-debt)).
