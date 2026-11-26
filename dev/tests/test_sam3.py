"""The SAM3 caller, on generated media (CPU; the job is faked, so no GPU and no SAM3 environment)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from conftest import SIZE

from rin import Master, Run, core
from rin.servants import sam3

SAM3_MODULE = sys.modules["rin.servants.sam3"]


@pytest.fixture
def frames(m: Master, clips: Path) -> Run:
    return core.sample(m, core.ingest(m, clips, split="test"), fps=4)


@pytest.fixture
def checkpoint(workspace: Path) -> Path:
    path = workspace / "checkpoints" / "sam3.pt"
    path.parent.mkdir()
    path.write_bytes(b"not really weights")
    return path


def _fake_job(*, per_image: int = 2, masks: bool = True) -> object:
    """Stands in for job.py: on every image, `per_image` boxes in pixels: the left half of the
    image, then the top-left quarter, ..., each with a made-up mask."""

    def run(_job_dir: Path, spec_file: Path) -> None:
        spec = json.loads(spec_file.read_text())
        found = []
        for i, image in enumerate(spec["images"]):
            for k in range(per_image):
                found.append({
                    "image": i, "class": "person", "score": 0.5 + 0.1 * k,
                    "box": [0.0, 0.0, image["width"] / 2, image["height"] / (k + 1)],
                    "mask_rle": "fake-rle" if masks else None,
                })
        (spec_file.parent / "job.json").write_text(json.dumps({
            "detections": found, "versions": {"torch": "fake", "pycocotools": "fake"},
            "seconds": 1.0, "peak_vram_gib": 2.0,
        }))

    return run


@pytest.fixture(autouse=True)
def fixed_engine_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SAM3_MODULE, "vendor_commit", lambda _repo: "engine123")


def test_concept_files_the_jobs_boxes_in_unit_coordinates(
    m: Master, frames: Run, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SAM3_MODULE, "run_job", _fake_job())
    run = sam3.concept(m, frames, prompts={"person": "person"}, checkpoint=checkpoint)

    df = run.instances().df().merge(m.sql("SELECT media_id, path, width FROM media").df(), on="media_id")
    assert len(df) == 2 * len(frames) and list(df["instance_idx"]) == list(range(len(df)))
    assert set(map(tuple, df[["media_id", "frame_idx"]].values.tolist())) == set(
        map(tuple, frames.frame_set().df()[["media_id", "frame_idx"]].values.tolist())
    )
    assert (df["x1"] == 0.5).all() and (df["x0"] == 0).all()  # half the width, whatever the media's size
    assert sorted(set(df["y1"])) == [0.5, 1.0]
    assert (df["class"] == "person").all() and (df["mask_rle"] == "fake-rle").all()
    assert set(df.loc[df["path"].str.endswith(".jpg"), "width"]) == {48} and set(df["width"]) >= {SIZE[0]}

    info = run.info
    assert info["model"] == "sam3" and info["checkpoint"] == "checkpoints/sam3.pt"
    assert info["checkpoint_sha256"] == hashlib.sha256(b"not really weights").hexdigest()
    assert info["code"]["sam3_engine"] == "engine123" and info["code"]["torch"] == "fake"
    assert run.config == {"prompts": {"person": "person"}, "score_floor": 0.1, "masks": True, "images_per_session": 8}
    assert sorted(p.name for p in m.layout.run_dir(run.id).iterdir() if p.name != "src.diff") == [
        "instances.parquet", "run.json"
    ]


def test_concept_without_masks_stores_none(
    m: Master, frames: Run, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SAM3_MODULE, "run_job", _fake_job(masks=False))
    run = sam3.concept(m, frames, prompts={"person": "person"}, checkpoint=checkpoint, masks=False)
    assert run.instances().df()["mask_rle"].isna().all()


def test_a_missing_mask_is_refused(m: Master, frames: Run, checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SAM3_MODULE, "run_job", _fake_job(masks=False))
    with pytest.raises(RuntimeError, match="without a mask"):
        sam3.concept(m, frames, prompts={"person": "person"}, checkpoint=checkpoint)
    assert not any(m.layout.tmp.iterdir())


def test_a_frame_with_nothing_found_is_fine(
    m: Master, frames: Run, checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SAM3_MODULE, "run_job", _fake_job(per_image=0))
    run = sam3.concept(m, frames, prompts={"person": "person"}, checkpoint=checkpoint)
    assert run.instances().df().empty


def test_a_failed_job_leaves_no_run(m: Master, frames: Run, checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(_job_dir: Path, _spec: Path) -> None:
        raise RuntimeError("The sam3 job failed (exit code 1)")

    monkeypatch.setattr(SAM3_MODULE, "run_job", crash)
    before = {r for (r,) in m.sql("SELECT run_id FROM runs").fetchall()}
    with pytest.raises(RuntimeError, match="job failed"):
        sam3.concept(m, frames, prompts={"person": "person"}, checkpoint=checkpoint)
    assert {r for (r,) in m.sql("SELECT run_id FROM runs").fetchall()} == before
    assert not any(m.layout.tmp.iterdir())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"prompts": {}}, "prompts"),
        ({"prompts": {"person": ["person", "pedestrian"]}}, "prompts"),
        ({"prompts": {"": "person"}}, "prompts"),
        ({"score_floor": 1.0}, "score_floor"),
        ({"score_floor": True}, "score_floor"),
        ({"masks": 1}, "masks"),
        ({"images_per_session": 0}, "images_per_session"),
    ],
)
def test_bad_arguments_fail_before_any_work(
    m: Master, frames: Run, checkpoint: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict, error: str
) -> None:
    def no_job(*_args: object) -> None:
        raise AssertionError("the job ran although the arguments were bad")

    monkeypatch.setattr(SAM3_MODULE, "run_job", no_job)
    args = {"prompts": {"person": "person"}, "checkpoint": checkpoint, **kwargs}
    with pytest.raises(ValueError, match=error):
        sam3.concept(m, frames, **args)


def test_a_missing_checkpoint_is_refused(m: Master, frames: Run, workspace: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        sam3.concept(m, frames, prompts={"person": "person"}, checkpoint=workspace / "nope.pt")
