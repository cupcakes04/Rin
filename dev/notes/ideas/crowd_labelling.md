# Crowds: the labelling cost, not the selection

**Status:** raw (the user, 2026-11-27, in chat after M4). Kept apart from the holy grail on purpose: the user's own words were "don't let it bias our reality".

**The pain.** About 95% of the user's labelling stress in M3 was crowds: dense groups where nobody can box each person, now marked as `crowd` regions that scoring ignores (D-34). The user's work may involve crowds too; whether that matters for Rin's direction is open.

**Why it is a different problem.** The holy grail asks *which* frames to label; crowds make *each* frame expensive to label, whichever is chosen. Selection can't fix that. What could:
- **Truth someone else paid for:** public crowd datasets labelled by people, as a checked test set, so no one here boxes crowds by hand. Candidates to check (not yet looked at: licence, video or stills, fixed cameras or not): MOT20, CrowdHuman.
- **Review by exception** ([review_by_exception.md](review_by_exception.md)): a person checks only what the teacher is unsure of.
- **Crowds as ignore regions in training too** ([parked debt](../milestones/README.md#parked-debt), "A student may learn better with crowds as ignore regions").

**Note.** The gym's real headroom may hide in its crowds: scoring ignores crowd regions (D-34), and stock detectors are weakest in occlusion. Measuring that needs crowd truth, which is exactly what is missing.
