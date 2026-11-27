"""yolo.train's caller on a real export, with the job faked (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_export import setup  # noqa: F401 (a fixture)

from rin import Master, Run, export
from rin.servants import yolo

YOLO_MODULE = sys.modules["rin.servants.yolo"]


@pytest.fixture
def exported(m: Master, setup: dict[str, Run]) -> Run:  # noqa: F811
    train = m.dataset(labels=[setup["labels"]], frames=setup["pool"], top=8, gate="score >= 0.5")
    val = m.dataset(labels=[setup["labels"]], frames=setup["val"], gate="score >= 0.5")
    return export.yolo(m, train, classes=["person"], val=val)


@pytest.fixture
def checkpoint(workspace: Path) -> Path:
    path = workspace / "checkpoints" / "yolo" / "yolo26s.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"stock weights")
    return path


def _fake_job(seen: dict) -> object:
    """Stands in for job.py: checks it was handed a copy of the export, writes Ultralytics' layout."""

    def run(_job_dir: Path, spec_file: Path) -> None:
        spec = json.loads(spec_file.read_text())
        seen.update(spec)
        data = Path(spec["data"])
        assert data.read_text().startswith("train: images/train")
        (data.parent / "labels" / "train.cache").write_text("Ultralytics writes this")  # into the copy
        out = spec_file.parent / "train"
        (out / "weights").mkdir(parents=True)
        for name in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"):
            (out / name).write_text(name)
        (spec_file.parent / "job.json").write_text(json.dumps({
            "save_dir": str(out), "best_fitness": 0.4, "seconds": 120.0, "peak_vram_gib": 5.5,
            "metrics": {"metrics/precision(B)": 0.8, "metrics/recall(B)": 0.7, "metrics/mAP50(B)": 0.75,
                        "metrics/mAP50-95(B)": 0.45, "val/box_loss": 1.2},
            "versions": {"torch": "fake", "ultralytics": "fake"},
        }))

    return run


def test_train_files_weights_metrics_and_leaves_the_export_untouched(
    m: Master, exported: Run, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = sorted(p.relative_to(m.layout.run_dir(exported.id)) for p in m.layout.run_dir(exported.id).rglob("*"))
    seen: dict = {}
    monkeypatch.setattr(YOLO_MODULE, "run_job", _fake_job(seen))
    run = yolo.train(m, exported, model=checkpoint, epochs=4, imgsz=640, batch=8, seed=1, tags={"n": 8})

    folder = m.layout.run_dir(run.id)
    assert sorted(p.name for p in (folder / "files").iterdir()) == ["args.yaml", "best.pt", "last.pt", "results.csv"]
    assert not (folder / "data").exists() and not (folder / "train").exists() and not (folder / "spec.json").exists()
    metrics = dict(m.sql("SELECT name, value FROM metrics WHERE run_id = ?", [run.id]).fetchall())
    assert metrics["val/mAP50-95"] == 0.45 and metrics["seconds_per_epoch"] == 30.0
    assert dict(run.table("artifacts").fetchall())["best"] == "files/best.pt"
    assert run.info["model"] == "yolo26s" and run.info["checkpoint"] == "checkpoints/yolo/yolo26s.pt"
    assert len(run.info["checkpoint_sha256"]) == 64
    assert seen["epochs"] == 4 and seen["seed"] == 1 and run.tags == {"n": 8}
    # The optimizer is stated (D-40): what Ultralytics' "auto" picked for M4, one step per batch.
    stated = {"optimizer": "AdamW", "lr0": 0.002, "momentum": 0.9, "warmup_bias_lr": 0.0, "accumulate": 1}
    assert {k: seen[k] for k in stated} == stated and {k: run.config[k] for k in stated} == stated
    after = sorted(p.relative_to(m.layout.run_dir(exported.id)) for p in m.layout.run_dir(exported.id).rglob("*"))
    assert after == before  # the labels cache went into the copy, not the export


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"imgsz": 650}, "multiple of 32"), ({"epochs": 0}, "epochs"), ({"batch": True}, "batch"),
     ({"optimizer": "auto"}, "never 'auto'"), ({"lr0": 0}, "lr0 must be above 0"), ({"momentum": 1.0}, "momentum"),
     ({"accumulate": 0}, "accumulate")],
)
def test_train_checks_arguments(m: Master, exported: Run, checkpoint: Path, kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        yolo.train(m, exported, **{"model": checkpoint, "epochs": 2, "imgsz": 640, "batch": 4, "seed": 0, **kwargs})


def test_train_needs_an_export(m: Master, exported: Run, checkpoint: Path) -> None:
    with pytest.raises(TypeError, match="export.yolo"):
        yolo.train(m, m.run(exported.inputs[0]), model=checkpoint, epochs=2, imgsz=640, batch=4, seed=0)


def _fake_predict(seen: dict, detections: list[dict]) -> object:
    """Stands in for job.py's predict: records the spec, returns the given detections."""

    def run(_job_dir: Path, spec_file: Path) -> None:
        seen.update(json.loads(spec_file.read_text()))
        (spec_file.parent / "job.json").write_text(json.dumps({
            "detections": detections, "seconds": 1.0, "versions": {"torch": "fake", "ultralytics": "fake"},
        }))

    return run


def test_predict_hands_classes_to_the_job_and_clamps_rounding(
    m: Master, setup: dict[str, Run], checkpoint: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    seen: dict = {}
    dets = [{"image": 0, "class": "person", "score": 0.8, "box": [-1e-7, 0.1, 0.5, 1.0000001]},
            {"image": 1, "class": "person", "score": 0.4, "box": [0.2, 0.2, 0.4, 0.6]}]
    monkeypatch.setattr(YOLO_MODULE, "run_job", _fake_predict(seen, dets))
    run = yolo.predict(m, checkpoint, setup["val"], imgsz=1280, classes=["person"])
    assert seen["classes"] == ["person"] and seen["max_det"] == 100
    boxes = run.instances().df().sort_values("score", ascending=False)
    assert boxes[["x0", "y1"]].values.tolist()[0] == [0.0, 1.0]
    assert run.info["model"] == "yolo26s" and run.info["checkpoint"] == "checkpoints/yolo/yolo26s.pt"


@pytest.mark.parametrize(
    ("box", "cls", "error"),
    [([0.1, 0.1, 1.001, 0.5], "person", "outside \\[0, 1\\]"), ([0.1, 0.1, 0.3, 0.5], "chair", "not asked for")],
)
def test_predict_refuses_what_the_job_should_not_return(
    m: Master, setup: dict[str, Run], checkpoint: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
    box: list, cls: str, error: str,
) -> None:
    monkeypatch.setattr(YOLO_MODULE, "run_job", _fake_predict({}, [{"image": 0, "class": cls, "score": 0.5, "box": box}]))
    with pytest.raises((ValueError, RuntimeError), match=error):
        yolo.predict(m, checkpoint, setup["val"], imgsz=1280, classes=["person"])


@pytest.mark.parametrize(
    ("kwargs", "error"), [({"imgsz": 650}, "multiple of 32"), ({"classes": ["person", "person"]}, "twice")],
)
def test_predict_checks_arguments(m: Master, setup: dict[str, Run], checkpoint: Path, kwargs: dict, error: str) -> None:  # noqa: F811
    with pytest.raises(ValueError, match=error):
        yolo.predict(m, checkpoint, setup["val"], **{"imgsz": 640, **kwargs})
