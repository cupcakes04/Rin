"""M4 step 6b: does a small budget gain from more optimizer steps? (epoch_images, accumulate; D-40)

Per-camera random at seed 0 on M4's candidates; each budget trained with its frames repeated so an
epoch shows at least K images, plus M4's setting (one step per 8 batches) on the same frames as a
control; every student scored on the checked test set. Resumable: a run whose inputs and config
match one already in the store is reused, not redone.
"""
import json
import time

from rin import Master, curate, evaluate, export
from rin.servants import yolo

m = Master("workspace/store")
labels = m.run("20261127-111139-sam3.concept-5477")
checked = m.run("20261127-094340-human.annotate-64f9")
truth = m.run("20261127-152506-export.coco-8bc8")
cands = m.run("20261127-105927-curate.dedupe-6b65")
val = m.dataset(labels=[labels], frames=m.run("20261127-110156-curate.dedupe-f05d"), gate="score >= 0.5", duplicate_iou=0.7)
CHECKPOINT = "workspace/checkpoints/yolo/yolo26s.pt"


def existing(step: str, inputs: list[str], **config) -> str | None:
    """The latest run of `step` on exactly these inputs whose config holds these values."""
    for run_id, cfg in m.sql("SELECT run_id, config FROM runs WHERE step = ? AND inputs = ? ORDER BY run_id DESC",
                             [step, inputs]).fetchall():
        cfg = json.loads(cfg)
        if all(cfg.get(k) == v for k, v in config.items()):
            return run_id
    return None


sel_id = existing("curate.select", [cands.id], method="random", n_max=1000, seed=0)
picks = m.run(sel_id) if sel_id else curate.select(m, cands, None, method="random", n_max=1000, seed=0)
print("SELECT", picks.id, flush=True)

# (N, epoch_images, accumulate); M4's setting was accumulate 8, no repeat
PLAN = [(25, None, 1), (100, None, 1), (25, 250, 1), (25, None, 8), (100, None, 8),
        (100, 250, 1), (25, 1000, 1), (100, 1000, 1), (1000, None, 1)]
exports = {}
for n, k, acc in PLAN:
    tags = {"step": "m4-6b", "method": "random", "seed": 0, "n": n, "epoch_images": k, "accumulate": acc}
    if n not in exports:
        ds = m.dataset(labels=[labels], frames=picks, top=n, gate="score >= 0.5", duplicate_iou=0.7)
        exp_id = next((r for r, t in m.sql("SELECT run_id, tags FROM runs WHERE step = 'export.yolo' ORDER BY run_id DESC").fetchall()
                       if json.loads(t or "{}") == {"step": "m4-6b", "n": n}), None)
        exports[n] = m.run(exp_id) if exp_id else export.yolo(m, ds, classes=["person"], val=val, tags={"step": "m4-6b", "n": n})
    t = time.time()
    train_id = existing("yolo.train", [exports[n].id], epochs=50, imgsz=1280, batch=8, seed=0, epoch_images=k, accumulate=acc)
    student = m.run(train_id) if train_id else yolo.train(
        m, exports[n], model=CHECKPOINT, epochs=50, imgsz=1280, batch=8, seed=0, epoch_images=k, accumulate=acc, tags=tags)
    pred_id = existing("yolo.predict", [student.id, checked.id], imgsz=1280, max_det=100, conf=0.001)
    preds = m.run(pred_id) if pred_id else yolo.predict(
        m, student, checked, conf=0.001, max_det=100, imgsz=1280, classes=["person"], tags=tags)
    ev_id = existing("evaluate.coco", [preds.id, truth.id])
    ev = m.run(ev_id) if ev_id else evaluate.coco(m, preds, truth, tags=tags)
    scores = dict(m.sql("SELECT name, round(value, 4) FROM metrics WHERE run_id = ?", [ev.id]).fetchall())
    secs = dict(m.sql("SELECT name, value FROM metrics WHERE run_id = ?", [student.id]).fetchall())["seconds"]
    print("RESULT", {"n": n, "epoch_images": k, "accumulate": acc}, exports[n].id, student.id, preds.id, ev.id,
          f"train {secs / 60:.1f} min", scores, flush=True)
