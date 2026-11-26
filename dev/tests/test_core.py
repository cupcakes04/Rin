"""rin.core: ingest and sample, end to end on generated media."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from conftest import VFR_STARTS_MS

from rin import Master, core


def _describe(path: Path) -> dict:
    return {"site": "lab", "camera": path.stem.upper(), "started_at": datetime(2018, 3, 7, 9, 55)}


def test_ingest_records_every_file_and_frame(m: Master, clips: Path) -> None:
    run = core.ingest(m, clips, split="pool", describe=_describe)

    media = run.media().df().set_index("path")
    assert sorted(media.index) == ["media/clips/cfr.mkv", "media/clips/still.jpg", "media/clips/vfr.mkv"]
    assert media.loc["media/clips/vfr.mkv", "n_frames"] == len(VFR_STARTS_MS)
    assert media.loc["media/clips/cfr.mkv", "n_frames"] == 16
    assert media.loc["media/clips/still.jpg", ["kind", "n_frames", "width"]].tolist() == ["image", 1, 48]
    assert set(media["split"]) == {"pool"}
    assert media.loc["media/clips/vfr.mkv", "camera"] == "VFR"

    vfr_id = media.loc["media/clips/vfr.mkv", "media_id"]
    pts = m.sql("SELECT pts FROM frames WHERE media_id = ? ORDER BY frame_idx", [vfr_id]).df()["pts"]
    np.testing.assert_allclose(pts, np.array(VFR_STARTS_MS) / 1000, atol=1e-6)
    assert run.config == {"source": "media/clips", "split": "pool", "describe": "_describe"}


def test_ingest_without_describe_leaves_camera_and_time_empty(m: Master, clips: Path) -> None:
    run = core.ingest(m, clips / "cfr.mkv", split="sample")
    assert run.media().df()[["site", "camera", "started_at"]].isna().all().all()


def test_ingesting_the_same_file_twice_is_refused(m: Master, clips: Path) -> None:
    core.ingest(m, clips, split="pool")
    with pytest.raises(ValueError, match="Already ingested"):
        core.ingest(m, clips / "cfr.mkv", split="test")


def test_media_id_comes_from_the_file_so_a_fresh_store_gives_the_same_id(workspace: Path, clips: Path) -> None:
    """Judgments point at media_id; rebuilding the store must not orphan them."""
    first = core.ingest(Master(workspace / "store"), clips / "cfr.mkv", split="pool").media().df()
    again = core.ingest(Master(workspace / "store_again"), clips / "cfr.mkv", split="pool").media().df()
    assert first["media_id"].tolist() == again["media_id"].tolist()
    assert first["media_id"][0].startswith("m") and len(first["media_id"][0]) == 17


def test_the_same_content_under_another_path_is_refused(m: Master, clips: Path) -> None:
    core.ingest(m, clips / "cfr.mkv", split="pool")
    shutil.copy(clips / "cfr.mkv", clips / "copy_of_cfr.mkv")
    with pytest.raises(ValueError, match="already ingested as media/clips/cfr.mkv"):
        core.ingest(m, clips / "copy_of_cfr.mkv", split="test")


def test_two_copies_in_one_ingest_are_refused(m: Master, clips: Path) -> None:
    shutil.copy(clips / "cfr.mkv", clips / "copy_of_cfr.mkv")
    with pytest.raises(ValueError, match="same file"):
        core.ingest(m, clips, split="pool")
    assert m.sql("SELECT count(*) FROM media").fetchall()[0] == (0,)


def test_media_outside_the_workspace_is_refused(m: Master, tmp_path: Path, clips: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "cfr.mkv").write_bytes((clips / "cfr.mkv").read_bytes())
    with pytest.raises(ValueError, match="outside the workspace"):
        core.ingest(m, outside, split="pool")


@pytest.mark.parametrize(
    ("returned", "error"),
    [
        ({"room": "a"}, "unknown keys"),
        ({"camera": 3}, "must be str"),
        (["camera"], "must return a dict"),
        ({"started_at": datetime(2018, 3, 7, 9, 55, tzinfo=timezone.utc)}, "no timezone"),
    ],
)
def test_bad_describe_output_is_refused(m: Master, clips: Path, returned: object, error: str) -> None:
    with pytest.raises((ValueError, TypeError), match=error):
        core.ingest(m, clips, split="pool", describe=lambda _p: returned)
    assert not any(m.layout.runs.iterdir())


def test_bad_describe_fails_before_any_file_is_decoded(
    m: Master, clips: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_decoding(path: Path) -> None:
        raise AssertionError(f"decoded {path.name} before describe was checked on every file")

    def breaks_on_last_file(path: Path) -> dict:  # files go in name order; vfr.mkv is last
        return {"camera": 3 if path.name == "vfr.mkv" else "ok"}

    # The module, not the function `rin.core.ingest` that shadows its name in the package.
    monkeypatch.setattr(sys.modules["rin.core.ingest"], "probe", no_decoding)
    with pytest.raises(TypeError, match="must be str"):
        core.ingest(m, clips, split="pool", describe=breaks_on_last_file)


def test_ingest_refuses_empty_split_and_empty_folder(m: Master, workspace: Path) -> None:
    empty = workspace / "media" / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="split"):
        core.ingest(m, empty, split="")
    with pytest.raises(ValueError, match="No video or image"):
        core.ingest(m, empty, split="pool")


def test_sample_picks_the_frames_on_screen(m: Master, clips: Path) -> None:
    pool = core.ingest(m, clips, split="pool")
    frames = core.sample(m, pool, fps=4)

    picked = frames.frame_set().df().merge(pool.media().df()[["media_id", "path"]], on="media_id")
    by_path = {p: sorted(g["frame_idx"]) for p, g in picked.groupby("path")}
    # CFR 10 fps, every 0.25 s: frames starting at 0.0, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5.
    assert by_path["media/clips/cfr.mkv"] == [0, 2, 5, 7, 10, 12, 15]
    # VFR: 0 s -> 0, 0.25 s -> 7 (starts 0.233), 0.5 s -> 12 (0.4), then one new frame per 0.25 s.
    assert by_path["media/clips/vfr.mkv"] == [0, 7, *range(12, 31)]
    assert by_path["media/clips/still.jpg"] == [0]

    assert frames.inputs == [pool.id] and frames.config == {"fps": 4}
    assert len(frames) == 7 + 21 + 1


def test_sample_needs_an_ingest_run(m: Master, clips: Path) -> None:
    frames = core.sample(m, core.ingest(m, clips, split="pool"), fps=1)
    with pytest.raises(TypeError, match="needs an ingest run"):
        core.sample(m, frames, fps=1)


def test_len_of_a_run_without_a_frame_set_is_an_error(m: Master, clips: Path) -> None:
    with pytest.raises(TypeError, match="no frame set"):
        len(core.ingest(m, clips, split="pool"))


def test_sample_refuses_a_bool_as_fps(m: Master, clips: Path) -> None:
    pool = core.ingest(m, clips, split="pool")
    with pytest.raises(ValueError, match="fps"):
        core.sample(m, pool, fps=True)


def test_ingest_takes_a_list_of_files(m: Master, clips: Path) -> None:
    run = core.ingest(m, [clips / "cfr.mkv", clips / "still.jpg"], split="test")
    assert sorted(run.media().df()["path"]) == ["media/clips/cfr.mkv", "media/clips/still.jpg"]
    assert run.config["source"] == ["media/clips/cfr.mkv", "media/clips/still.jpg"]
    with pytest.raises(ValueError, match="twice"):
        core.ingest(m, [clips / "vfr.mkv", clips / "vfr.mkv"], split="test")
    with pytest.raises(FileNotFoundError):
        core.ingest(m, [clips / "nope.mkv"], split="test")

