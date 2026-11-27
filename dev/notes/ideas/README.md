# Ideas

One file per idea. Status moves **raw → explored → adopted** (it becomes a decision or a milestone) or **rejected** (the file says why, briefly, then gets deleted once that no longer matters).

| Idea | Status | One line |
|---|---|---|
| [holy_grail.md](holy_grail.md) | adopted (D-04, D-17) | Choose what to label from raw video, and prove it helped |
| [vlm_window_actions.md](vlm_window_actions.md) | explored (D-05) | Actions as VLM window classification, distilled into a DINOv3 head |
| [tracker_tuning.md](tracker_tuning.md) | raw | Tune BoxMOT against SAM3 tracks; one config per edge deployment |
| [actors.md](actors.md) | raw | People first, generic tracks for everything |
| [search.md](search.md) | raw | Text and example search over footage |
| [crowd_labelling.md](crowd_labelling.md) | raw | Crowds make each frame expensive to label; a labelling-cost problem (public truth, review by exception), kept apart from selection |
| [where_distillation_pays.md](where_distillation_pays.md) | adopted (D-17) | Distil where the stock student is weak (new concepts, rare events, odd domains), not where COCO already works; the agent's M5 recommendation: a rare concept COCO lacks |
| [review_by_exception.md](review_by_exception.md) | raw, first measurement | A person checks only what the teacher is unsure of, by a rule measured on checked frames; per task, the teacher's measured reliability picks the workflow |

New thoughts go to the inbox, [../brainstorms/brainstorm.md](../brainstorms/brainstorm.md), first. The agent turns them into files here.
