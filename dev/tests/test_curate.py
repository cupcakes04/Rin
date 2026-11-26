"""rin.curate, the frame cache and the DINOv3 caller, on generated media (CPU; the job is faked)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
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

    def run(_job_dir: Path, spec_file: Path) -> None:
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
    monkeypatch.setattr(DINO_MODULE, "run_job", _fake_job(group=4))
    emb = dinov3.embed(m, every_frame, model="facebook/dinov3-test", what=["cls", "mean_patch"])

    n = len(every_frame)
    assert emb.vectors().shape == (2 * n, 32)
    index = emb.embeddings().df()
    assert list(index["row"]) == list(range(2 * n))
    assert list(index["kind"]) == ["cls"] * n + ["mean_patch"] * n
    files = sorted(p.name for p in m.layout.run_dir(emb.id).iterdir() if p.name != "src.diff")
    assert files == ["embeddings.npy", "embeddings.parquet", "run.json"]
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
    def crash(_job_dir: Path, _spec: Path) -> None:
        raise RuntimeError("The DINOv3 job failed (exit code 1)")

    monkeypatch.setattr(DINO_MODULE, "run_job", crash)
    before = {r for (r,) in m.sql("SELECT run_id FROM runs").fetchall()}
    with pytest.raises(RuntimeError, match="job failed"):
        dinov3.embed(m, every_frame, model="facebook/dinov3-test")
    assert {r for (r,) in m.sql("SELECT run_id FROM runs").fetchall()} == before
    assert not any(m.layout.tmp.iterdir())


def test_dedupe_drops_frames_like_the_last_kept_one(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DINO_MODULE, "run_job", _fake_job(group=4))
    emb = dinov3.embed(m, every_frame, model="facebook/dinov3-test")
    cands = curate.dedupe(m, every_frame, emb, cosine_max=0.95)

    df = cands.frame_set().df().merge(m.sql("SELECT media_id, path FROM media").df(), on="media_id")
    assert sorted(df.loc[df["path"].str.endswith("cfr.mkv"), "frame_idx"]) == [0, 4, 8, 12]
    assert cands.inputs == [every_frame.id, emb.id]


def test_dedupe_needs_a_vector_for_every_frame(m: Master, clips: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = core.ingest(m, clips, split="pool")
    monkeypatch.setattr(DINO_MODULE, "run_job", _fake_job(group=4))
    emb = dinov3.embed(m, core.sample(m, pool, fps=1), model="facebook/dinov3-test")
    with pytest.raises(ValueError, match="no 'cls' vector"):
        curate.dedupe(m, core.sample(m, pool, fps=10), emb, cosine_max=0.95)


# --- the core stays free of model frameworks -------------------------------------------------


def test_importing_rin_loads_no_model_framework() -> None:
    code = (
        "import sys, rin, rin.curate, rin.viewer, rin.human, rin.servants.frame_cache, rin.servants.dinov3, rin.servants.sam3\n"
        "loaded = [m for m in ('torch', 'transformers', 'torchvision', 'sam3', 'sam3_engine', 'pycocotools', 'fiftyone')"
        " if m in sys.modules]\n"
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vectors_with_nan_are_refused(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """DINOv3 in pure float16 overflowed to NaN on every frame; such a run must never be filed."""
    fake = _fake_job(group=4)

    def nan_job(job_dir: Path, spec_file: Path) -> None:
        fake(job_dir, spec_file)
        vectors = np.load(spec_file.parent / "embeddings.npy")
        vectors[3] = np.nan
        np.save(spec_file.parent / "embeddings.npy", vectors)

    monkeypatch.setattr(DINO_MODULE, "run_job", nan_job)
    with pytest.raises(RuntimeError, match="non-finite"):
        dinov3.embed(m, every_frame, model="facebook/dinov3-test")
    assert not any(m.layout.tmp.iterdir())


def test_center_removes_each_cameras_mean(m: Master, clips: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def camera_of(path: Path) -> dict:
        return {"camera": "A" if path.stem == "cfr" else "B"}

    frames = core.sample(m, core.ingest(m, clips, split="pool", describe=camera_of), fps=10)
    monkeypatch.setattr(DINO_MODULE, "run_job", _fake_job(group=4))
    emb = dinov3.embed(m, frames, model="facebook/dinov3-test", what=["cls", "mean_patch"])
    centred = curate.center(m, emb)

    index = centred.embeddings().df().merge(m.sql("SELECT media_id, camera FROM media").df(), on="media_id")
    before, after = emb.vectors().astype(np.float32), centred.vectors().astype(np.float32)
    for (_, _), rows in index.groupby(["camera", "kind"])["row"]:
        r = rows.to_numpy()
        np.testing.assert_allclose(after[r], before[r] - before[r].mean(axis=0), atol=1e-3)
        np.testing.assert_allclose(after[r].mean(axis=0), 0, atol=1e-3)
    assert list(index.sort_values("row")["kind"]) == list(emb.embeddings().df()["kind"])
    assert centred.inputs == [emb.id] and centred.info["model"] is None  # a rule; the model is on its input


def test_center_refuses_a_camera_with_one_frame(m: Master, clips: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One frame minus its own mean is a zero vector, which dedupe can't compare."""
    def camera_of(path: Path) -> dict:
        return {"camera": "alone" if path.suffix == ".jpg" else "A"}

    frames = core.sample(m, core.ingest(m, clips, split="pool", describe=camera_of), fps=10)
    monkeypatch.setattr(DINO_MODULE, "run_job", _fake_job(group=4))
    emb = dinov3.embed(m, frames, model="facebook/dinov3-test")
    with pytest.raises(ValueError, match="camera alone has a single 'cls' vector"):
        curate.center(m, emb)
    assert not any(m.layout.tmp.iterdir())


def test_dedupe_refuses_zero_vectors(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero vector used to become NaN and silently drop every later frame of its stream."""
    fake = _fake_job(group=4)

    def zero_job(job_dir: Path, spec_file: Path) -> None:
        fake(job_dir, spec_file)
        vectors = np.load(spec_file.parent / "embeddings.npy")
        vectors[0] = 0
        np.save(spec_file.parent / "embeddings.npy", vectors)

    monkeypatch.setattr(DINO_MODULE, "run_job", zero_job)
    emb = dinov3.embed(m, every_frame, model="facebook/dinov3-test")
    with pytest.raises(ValueError, match="all-zero"):
        curate.dedupe(m, every_frame, emb, cosine_max=0.9)


@pytest.mark.parametrize("bad", [True, False])
def test_bools_are_not_numbers(m: Master, every_frame: Run, bad: bool) -> None:
    with pytest.raises(ValueError, match="cosine_max"):
        curate.dedupe(m, every_frame, every_frame, cosine_max=bad)


def test_a_run_times_its_own_work(m: Master, every_frame: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """started_at is stamped before the step's work, so finished_at - started_at is what it cost."""
    scores_module = sys.modules["rin.curate.scores"]
    real = scores_module.read_frames

    def slow(*args: object, **kwargs: object):  # noqa: ANN202
        time.sleep(0.1)
        return real(*args, **kwargs)

    monkeypatch.setattr(scores_module, "read_frames", slow)
    run = curate.score_frames(m, every_frame, ["dark"])
    took = datetime.fromisoformat(run.info["finished_at"]) - datetime.fromisoformat(run.info["started_at"])
    assert took.total_seconds() >= 0.1 * 3  # three media, each read once


def test_an_empty_cached_frame_is_decoded_again(m: Master, every_frame: Run) -> None:
    first = cached_frames(m, every_frame)
    _, _, path = first[0]
    path.write_bytes(b"")  # what a crash before the data reached the disk could leave
    cached_frames(m, every_frame)
    assert path.stat().st_size > 0


def test_stray_partial_frames_are_reported(m: Master, every_frame: Run) -> None:
    media_id, _, path = cached_frames(m, every_frame)[0]
    (path.parent / "3.jpg.999.partial").write_bytes(b"x")
    with pytest.warns(UserWarning, match="unfinished frame file"):
        cached_frames(m, every_frame)


# --- random ----------------------------------------------------------------------------------


def test_random_draws_n_distinct_frames_in_a_ranked_order(m: Master, every_frame: Run) -> None:
    picked = curate.random(m, every_frame, n=10, seed=3)
    df = picked.frame_set().df()
    assert len(df) == 10 and sorted(df["rank"]) == list(range(10))
    assert not df.duplicated(["media_id", "frame_idx"]).any()
    every = set(map(tuple, every_frame.frame_set().df()[["media_id", "frame_idx"]].values.tolist()))
    assert set(map(tuple, df[["media_id", "frame_idx"]].values.tolist())) <= every
    assert picked.config == {"n": 10, "seed": 3} and picked.inputs == [every_frame.id]


def test_random_with_the_same_seed_picks_the_same_frames(m: Master, every_frame: Run) -> None:
    def order(run: Run) -> list:
        return run.frame_set().df().sort_values("rank")[["media_id", "frame_idx"]].values.tolist()

    assert order(curate.random(m, every_frame, n=5, seed=0)) == order(curate.random(m, every_frame, n=5, seed=0))
    assert order(curate.random(m, every_frame, n=5, seed=0)) != order(curate.random(m, every_frame, n=5, seed=1))


@pytest.mark.parametrize(("n", "seed", "error"), [(0, 0, "between 1"), (10_000, 0, "between 1"), (True, 0, "n must"), (5, -1, "seed")])
def test_random_refuses_bad_arguments(m: Master, every_frame: Run, n: object, seed: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        curate.random(m, every_frame, n=n, seed=seed)


# --- union -----------------------------------------------------------------------------------


def test_union_holds_every_frame_once(m: Master, clips: Path) -> None:
    a = core.sample(m, core.ingest(m, clips / "cfr.mkv", split="test"), fps=10)
    b = core.sample(m, core.ingest(m, clips / "vfr.mkv", split="test"), fps=4)
    a_again = curate.random(m, a, n=5, seed=0)  # overlaps a
    both = curate.union(m, [a, b, a_again])
    frames = both.frame_set().df()
    assert len(frames) == len(a) + len(b) and not frames.duplicated(["media_id", "frame_idx"]).any()
    assert frames["rank"].isna().all() and both.inputs == [a.id, b.id, a_again.id]


@pytest.mark.parametrize("bad", ["one", "twice"])
def test_union_refuses_one_set_or_the_same_set_twice(m: Master, every_frame: Run, bad: str) -> None:
    with pytest.raises(ValueError, match="at least two" if bad == "one" else "twice"):
        curate.union(m, [every_frame] if bad == "one" else [every_frame, every_frame])

