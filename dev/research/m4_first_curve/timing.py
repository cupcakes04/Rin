import sys
from rin import Master
from rin.servants import yolo
m = Master("workspace/store")
imgsz, batch, epochs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
run = yolo.train(m, m.run("20261127-150150-export.yolo-55bf"), model="workspace/checkpoints/yolo/yolo26s.pt",
                 epochs=epochs, imgsz=imgsz, batch=batch, seed=0, tags={"purpose": "timing"})
print("RESULT", run.id, dict(m.sql("SELECT name, round(value, 4) FROM metrics WHERE run_id = ?", [run.id]).fetchall()))
