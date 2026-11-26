# Audit prompt

For an outside agent, any model, started fresh. Give it this file.

---

You are auditing Rin, a project built by another agent working with its user. Read as widely as you can: everything in this repo, including the old version in `dev/rin_v1/` (start with `CLAUDE.md` and `dev/notes/state.md` for orientation). Then tell us what we are missing: bugs, lessons the old version learned that the new one lacks, problems with the experiment, doubts about the direction, anything else you find. Judge freely; nothing is off limits, including decisions already made.

Change nothing in the repo except one file: write your report to `dev/audits/<YYYY-MM-DD>.md`. For each finding, point to where you saw it (file and line, or the command you ran), so it can be checked quickly.
