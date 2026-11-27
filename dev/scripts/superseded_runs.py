"""The store's superseded runs, for a milestone's close: lists them; deletes them only with --delete.

    uv run python dev/scripts/superseded_runs.py            # list
    uv run python dev/scripts/superseded_runs.py --delete   # after the user's yes

A run is kept when the current notes, write-ups, scripts or docs cite it (full id, or a 4-character
suffix no other run shares), when it is a person's work or an ingest, when a kept run depends on
it, or when it is a prediction or score of a kept model (the numbers write-ups report). Audit
reports don't count: they are read once, at triage. Everything else is listed, newest first.
"""
import re
import sys
from pathlib import Path
from rin import Master

m = Master("workspace/store")
runs = {r: (step, inputs or []) for r, step, inputs in m.sql("SELECT run_id, step, inputs FROM runs").fetchall()}
text = ""
for root in ("dev/notes", "dev/research", "dev/scripts", "docs"):
    for p in Path(root).rglob("*"):
        if p.suffix in (".md", ".py", ".sh"):
            text += p.read_text()
text += Path("CLAUDE.md").read_text()
tokens = set(re.findall(r"\b[0-9a-f]{4}\b", text))
pinned = {r for r in runs if r in text or (r[-4:] in tokens and len([x for x in runs if x[-4:] == r[-4:]]) == 1)}
pinned |= {r for r, (step, _) in runs.items() if step.startswith("human.") or step == "core.ingest"}
keep, todo = set(), list(pinned)
while todo:
    r = todo.pop()
    if r in keep or r not in runs:
        continue
    keep.add(r)
    todo.extend(runs[r][1])
# A kept model's predictions and their scores hold the numbers the write-ups report: keep them too.
grew = True
while grew:
    extra = {r for r, (step, inputs) in runs.items() if r not in keep and step in ("yolo.predict", "evaluate.coco")
             and inputs and inputs[0] in keep and runs[inputs[0]][0] in ("yolo.train", "yolo.predict")}
    grew = bool(extra)
    todo = list(extra)
    while todo:
        r = todo.pop()
        if r not in keep:
            keep.add(r)
            todo.extend(runs[r][1])
gone = sorted(set(runs) - keep, reverse=True)  # newest first: children before parents
by_step = {}
for r in gone:
    by_step.setdefault(runs[r][0], []).append(r)
print(f"{len(runs)} runs, keep {len(keep)}, delete {len(gone)}")
for step, ids in sorted(by_step.items()):
    print(f"  {step:20s} {len(ids):3d}  {', '.join(i[-4:] for i in sorted(ids))}")
size = sum(f.stat().st_size for r in gone for f in m.layout.run_dir(r).rglob("*") if f.is_file())
print(f"frees {size / 2**30:.1f} GiB")
if "--delete" in sys.argv:
    for r in gone:
        m.delete(m.run(r))
    print("deleted", len(gone))
