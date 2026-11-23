# Brainstorms

The user writes freely in [brainstorm.md](brainstorm.md), the inbox. The agent ingests it at the start of a session (the loop is in [CLAUDE.md, "How we iterate"](../../CLAUDE.md#how-we-iterate)). Each ingested brainstorm leaves two files here:
- `raw_<date>.md`: the user's own words, unedited.
- `processed_<date>.md`: where each point went, with the agent's verdict.

Once all of a brainstorm's points have landed and the raw text is no longer needed, both files can be deleted; git keeps them.

| Date | Processed | Topic |
|---|---|---|
| 2026-11-23 | [processed_2026-11-23.md](processed_2026-11-23.md) | Core reset: why v1 failed, model roles, holy grail, roles |
