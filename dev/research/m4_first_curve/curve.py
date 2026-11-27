"""M4 step 5: COCO truth, stock baseline, one student per budget (random, seed 0)."""
import time
from rin import Master, evaluate, export
from rin.servants import yolo

m = Master("workspace/store")
labels = m.run("20261127-111139-sam3.concept-5477")
checked = m.run("20261127-094340-human.annotate-64f9")
truth = export.coco(m, checked, classes=["person"], crowd={"crowd": "person"})
print("TRUTH", truth.id, flush=True)

def score(model, tags, imgsz):
    preds = yolo.predict(m, model, checked, conf=0.001, max_det=100, imgsz=imgsz, classes=["person"], tags=tags)
    ev = evaluate.coco(m, preds, truth, tags=tags)
    print("SCORE", tags, preds.id, ev.id, dict(m.sql("SELECT name, round(value, 4) FROM metrics WHERE run_id = ?", [ev.id]).fetchall()), flush=True)

for imgsz in (640, 1280):
    score("workspace/checkpoints/yolo/yolo26s.pt", {"method": "stock", "n": 0, "imgsz": imgsz}, imgsz)

val = m.dataset(labels=[labels], frames=m.run("20261127-110156-curate.dedupe-f05d"), gate="score >= 0.5", duplicate_iou=0.7)
picks = m.run("20261127-132314-curate.select-5da9")
for n in (100, 250, 500, 1000):
    tags = {"method": "random", "seed": 0, "n": n}
    t = time.time()
    ds = m.dataset(labels=[labels], frames=picks, top=n, gate="score >= 0.5", duplicate_iou=0.7)
    exp = export.yolo(m, ds, classes=["person"], val=val, tags=tags)
    student = yolo.train(m, exp, model="workspace/checkpoints/yolo/yolo26s.pt", epochs=50, imgsz=1280, batch=8, seed=0, tags=tags)
    print("TRAINED", tags, exp.id, student.id, f"{(time.time() - t) / 60:.1f} min", flush=True)
    score(student, tags, 1280)
