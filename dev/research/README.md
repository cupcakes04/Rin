# Research

Write-ups of experiments: one Markdown file per question (e.g. "does curated selection beat random?", the holy grail proof in M5).

Each write-up holds:
- **the question** and what would count as an answer;
- **the setup**, as run ids from `workspace/store/`, so every number can be traced back to the runs that made it;
- **the results**: small tables and plots exported from the store (CSV, PNG), kept next to the write-up;
- **the verdict**, honest, including "it didn't work, and here is what we try next".

Raw runs, weights and frame caches stay in `workspace/` (heavy, gitignored). Only the conclusions and the small files that support them live here, in git.

## Write-ups

| Write-up | Question |
|---|---|
| [m2_filters_dedupe](m2_filters_dedupe/README.md) | What do cheap filters and DINOv3 de-duplication leave of the pool? |
| [m3_sam3_test_day](m3_sam3_test_day/README.md) | What does SAM3 find on the test day, and what does it cost? |
