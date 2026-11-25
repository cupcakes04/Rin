"""rin.curate, the frame cache and the DINOv3 caller, on generated media (CPU; the job is faked)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import GRAY_STEP
from PIL import Image

from rin import Master, Run, core, curate
from rin.servants.frame_cache import cached_frames
from rin.servants import dinov3

DINO_MODULE = sys.modules["rin.servants.dinov3"]


def _scores(m: Master, run: Run) -> dict[tuple[str, int, str], float]:
    df = run.frame_scores().df().merge(m.sql("SELECT media_id, path FROM media").df(), on="media_id")
    return {(Path(p).name, i, n): v for p, i, n, v in zip(df["path"], df["frame_idx"], df["name"], df["value"])}


@pytest.fixture
def every_frame(m: Master, clips: Path) -> Run:
    """The clips at 10 fps: every frame of cfr.mkv, gray level 8 * frame_idx."""
    return core.sample(m, core.ingest(m, clips, split="pool"), fps=10)


# --- score_frames and keep -------------------------------------------------------------------


def test_score_frames_measures_each_frame(m: Master, every_frame: Run) -> None:
    run = curate.score_frames(m, every_frame, ["motion", "blur", "dark", "bright"])
    s = _scores(m, run)
    step = GRAY_STEP / 255
    for i in range(16):  # each frame is 8 gray levels from the next; the first compares with the second
        assert s[("cfr.mkv", i, "motion")] == pytest.approx(step, abs=0.004)
        assert s[("cfr.mkv", i, "blur")] == pytest.approx(0.0, abs=1e-4)  # flat frames: no edges
        assert s[("cfr.mkv", i, "bright")] == 0.0
    assert [s[("cfr.mkv", i, "dark")] for i in (0, 1, 3, 15)] == [1.0, 1.0, 0.0, 0.0]
    assert s[("still.jpg", 0, "dark")] == 1.0
    assert ("still.jpg", 0, "motion") not in s  # one frame: nothing to compare with
    assert run.config["names"] == ["motion", "blur", "dark", "bright"] and run.inputs == [every_frame.id]


def test_score_frames_refuses_unknown_scores(m: Master, every_frame: Run) -> None:
    with pytest.raises(ValueError, match="names must be"):
        curate.score_frames(m, every_frame, ["sharpness"])


def test_keep_applies_the_where_text_to_stored_scores(m: Master, every_frame: Run) -> None:
    scores = curate.score_frames(m, every_frame, ["motion", "dark"])
    kept = curate.keep(m, scores, where="dark < 0.5 AND motion > 0")
    df = kept.frame_set().df().merge(m.sql("SELECT media_id, path FROM media").df(), on="media_id")
    cfr = sorted(df.loc[df["path"].str.endswith("cfr.mkv"), "frame_idx"])
    assert cfr[0] >= 2 and cfr[-1] == 15 and 0 not in cfr and 1 not in cfr
    assert not df["path"].str.endswith("still.jpg").any()  # no motion score, so it can't pass
    assert kept.config == {"where": "dark < 0.5 AND motion > 0"}


def test_keep_refuses_a_where_nothing_passes(m: Master, every_frame: Run) -> None:
    scores = curate.score_frames(m, every_frame, ["dark"])
    with pytest.raises(ValueError, match="No frame"):
        curate.keep(m, scores, where="dark > 2")


# --- frame cache -----------------------------------------------------------------------------


def test_frames_are_cached_once_with_the_right_pixels(
    m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = cached_frames(m, every_frame)
    assert len(first) == len(every_frame) and all(p.is_file() for _, _, p in first)
    cfr_id = m.sql("SELECT media_id FROM media WHERE path LIKE '%cfr.mkv'").fetchall()[0][0]
    for media_id, i, path in first:
        if media_id == cfr_id:  # frame i of cfr.mkv is solid gray 8 * i
            with Image.open(path) as image:
                assert image.convert("RGB").getpixel((0, 0))[0] == pytest.approx(GRAY_STEP * i, abs=2)

    def no_decoding(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("decoded again although every frame was cached")

    monkeypatch.setattr(sys.modules["rin.servants.frame_cache"], "read_frames", no_decoding)
    assert cached_frames(m, every_frame) == first


# --- dinov3.embed (caller only; the job is faked) and dedupe ---------------------------------


def _fake_job(group: int) -> object:
    """Stands in for job.py: the vector of a frame is one-hot of (gray level // 8 // group), read
    back from the cached JPEG, so frames in the same group of `group` frames are identical."""

    def run(spec_file: Path) -> None:
        spec = json.loads(spec_file.read_text())
        rows = []
        for kind in spec["what"]:
            for p in spec["images"]:
                with Image.open(p) as image:
                    level = round(image.convert("L").getpixel((0, 0)) / GRAY_STEP)
                v = np.zeros(32, np.float16)
                v[(level // group) % 32] = 1 if kind == "cls" else -1
                rows.append(v)
        np.save(spec_file.parent / "embeddings.npy", np.stack(rows))
        (spec_file.parent / "job.json").write_text(json.dumps(
            {"model_revision": "abc123", "versions": {"torch": "fake", "transformers": "fake"}}
        ))

    return run


def test_embed_files_the_jobs_vectors_as_a_run(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DINO_MODULE, "_run_job", _fake_job(group=4))
    emb = dinov3.embed(m, every_frame, model="facebook/dinov3-test", what=["cls", "mean_patch"])

    n = len(every_frame)
    assert emb.vectors().shape == (2 * n, 32)
    index = emb.embeddings().df()
    assert list(index["row"]) == list(range(2 * n))
    assert list(index["kind"]) == ["cls"] * n + ["mean_patch"] * n
    assert sorted(p.name for p in m.layout.run_dir(emb.id).iterdir()) == ["embeddings.npy", "embeddings.parquet", "run.json"]
    assert emb.info["checkpoint"] == "abc123" and emb.info["code"]["transformers"] == "fake"
    assert emb.config == {"size": [896, 512], "what": ["cls", "mean_patch"], "batch_size": 32}


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"size": (900, 512)}, "multiple of 16"), ({"what": ["patches"]}, "what must be"), ({"batch_size": 0}, "batch_size")],
)
def test_embed_checks_arguments_before_any_work(m: Master, every_frame: Run, kwargs: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        dinov3.embed(m, every_frame, model="facebook/dinov3-test", **kwargs)
    assert not m.layout.frames.exists()


def test_a_failed_job_leaves_no_run(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(_spec: Path) -> None:
        raise RuntimeError("The DINOv3 job failed (exit code 1)")

    monkeypatch.setattr(DINO_MODULE, "_run_job", crash)
    before = {r for (r,) in m.sql("SELECT run_id FROM runs").fetchall()}
    with pytest.raises(RuntimeError, match="job failed"):
        dinov3.embed(m, every_frame, model="facebook/dinov3-test")
    assert {r for (r,) in m.sql("SELECT run_id FROM runs").fetchall()} == before
    assert not any(m.layout.tmp.iterdir())


def test_dedupe_drops_frames_like_the_last_kept_one(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DINO_MODULE, "_run_job", _fake_job(group=4))
    emb = dinov3.embed(m, every_frame, model="facebook/dinov3-test")
    cands = curate.dedupe(m, every_frame, emb, cosine_max=0.95)

    df = cands.frame_set().df().merge(m.sql("SELECT media_id, path FROM media").df(), on="media_id")
    assert sorted(df.loc[df["path"].str.endswith("cfr.mkv"), "frame_idx"]) == [0, 4, 8, 12]
    assert cands.inputs == [every_frame.id, emb.id]


def test_dedupe_needs_a_vector_for_every_frame(m: Master, clips: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = core.ingest(m, clips, split="pool")
    monkeypatch.setattr(DINO_MODULE, "_run_job", _fake_job(group=4))
    emb = dinov3.embed(m, core.sample(m, pool, fps=1), model="facebook/dinov3-test")
    with pytest.raises(ValueError, match="no 'cls' vector"):
        curate.dedupe(m, core.sample(m, pool, fps=10), emb, cosine_max=0.95)


# --- the core stays free of model frameworks -------------------------------------------------


def test_importing_rin_loads_no_model_framework() -> None:
    code = (
        "import sys, rin, rin.curate, rin.servants.frame_cache, rin.servants.dinov3\n"
        "loaded = [m for m in ('torch', 'transformers', 'torchvision') if m in sys.modules]\n"
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vectors_with_nan_are_refused(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """DINOv3 in pure float16 overflowed to NaN on every frame; such a run must never be filed."""
    fake = _fake_job(group=4)

    def nan_job(spec_file: Path) -> None:
        fake(spec_file)
        vectors = np.load(spec_file.parent / "embeddings.npy")
        vectors[3] = np.nan
        np.save(spec_file.parent / "embeddings.npy", vectors)

    monkeypatch.setattr(DINO_MODULE, "_run_job", nan_job)
    with pytest.raises(RuntimeError, match="non-finite"):
        dinov3.embed(m, every_frame, model="facebook/dinov3-test")
    assert not any(m.layout.tmp.iterdir())


def test_center_removes_each_cameras_mean(m: Master, clips: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def camera_of(path: Path) -> dict:
        return {"camera": "A" if path.stem == "cfr" else "B"}

    frames = core.sample(m, core.ingest(m, clips, split="pool", describe=camera_of), fps=10)
    monkeypatch.setattr(DINO_MODULE, "_run_job", _fake_job(group=4))
    emb = dinov3.embed(m, frames, model="facebook/dinov3-test", what=["cls", "mean_patch"])
    centred = curate.center(m, emb)

    index = centred.embeddings().df().merge(m.sql("SELECT media_id, camera FROM media").df(), on="media_id")
    before, after = emb.vectors().astype(np.float32), centred.vectors().astype(np.float32)
    for (_, _), rows in index.groupby(["camera", "kind"])["row"]:
        r = rows.to_numpy()
        np.testing.assert_allclose(after[r], before[r] - before[r].mean(axis=0), atol=1e-3)
        np.testing.assert_allclose(after[r].mean(axis=0), 0, atol=1e-3)
    assert list(index.sort_values("row")["kind"]) == list(emb.embeddings().df()["kind"])
    assert centred.inputs == [emb.id] and centred.info["model"] == "facebook/dinov3-test"
