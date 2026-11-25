"""rin.media: per-frame timestamps from real files, and time -> frame on screen."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
from conftest import (
    CFR_STARTS_MS,
    VFR_STARTS_MS,
    decoded_gray_ids,
    make_avi_with_bframes,
    make_clip,
    make_image,
    make_noisy_avi,
)

from rin.media import frame_on_screen, probe, sample_frames

# --- time -> frame ---------------------------------------------------------------------------


def test_frame_on_screen_is_the_latest_frame_started() -> None:
    pts = np.array([0.0, 1.0, 2.0])
    assert frame_on_screen(pts, [0.0, 0.99, 1.0, 1.5, 2.0, 99.0]).tolist() == [0, 0, 1, 1, 2, 2]


def test_frame_on_screen_absorbs_tick_rounding() -> None:
    # A frame meant to start at 0.1 s may read back as 0.1001 s.
    assert frame_on_screen(np.array([0.0, 0.1001, 0.2]), [0.1]).tolist() == [1]


def test_frame_on_screen_refuses_time_before_frame_0() -> None:
    with pytest.raises(ValueError, match="before frame 0"):
        frame_on_screen(np.array([0.0, 1.0]), [-0.5])


def test_frame_on_screen_refuses_unsorted_pts() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        frame_on_screen(np.array([0.0, 2.0, 1.0]), [0.5])


def test_sample_frames_halves_a_10fps_clip_at_5fps() -> None:
    pts = np.arange(16) / 10
    assert sample_frames(pts, 5).tolist() == [0, 2, 4, 6, 8, 10, 12, 14]


def test_sample_frames_above_the_video_rate_keeps_each_frame_once() -> None:
    pts = np.arange(4) / 10
    assert sample_frames(pts, 100).tolist() == [0, 1, 2, 3]


def test_sample_frames_refuses_bad_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        sample_frames(np.arange(4) / 10, 0)


# --- probe -----------------------------------------------------------------------------------


def test_probe_vfr_reads_real_timestamps(tmp_path: Path) -> None:
    info = probe(make_clip(tmp_path / "vfr.mkv", VFR_STARTS_MS, rate=30))
    assert (info.kind, info.width, info.height) == ("video", 64, 64)
    np.testing.assert_allclose(info.pts, np.array(VFR_STARTS_MS) / 1000, atol=1e-6)


def test_probe_cfr(tmp_path: Path) -> None:
    info = probe(make_clip(tmp_path / "cfr.mkv", CFR_STARTS_MS, rate=10))
    np.testing.assert_allclose(info.pts, np.arange(16) / 10, atol=1e-6)


def test_probe_avi_with_bframes_gets_display_order_times(tmp_path: Path) -> None:
    """MEVA's case: AVI attaches timestamps in storage order when B-frames are used."""
    path = make_avi_with_bframes(tmp_path / "b.avi", 20, rate=10)
    with av.open(str(path)) as container:
        raw = [f.time for f in container.decode(video=0)]
    assert not np.all(np.diff(raw) > 0), "precondition: this AVI must scramble its timestamps"

    info = probe(path)
    np.testing.assert_allclose(info.pts, np.arange(20) / 10, atol=1e-6)
    # And frame k really is the k-th frame shown, so frame_idx k <-> pts k/10.
    assert decoded_gray_ids(path) == list(range(20))


def test_probe_refuses_a_truncated_video(tmp_path: Path) -> None:
    """A clip cut in half still opens and decodes; only the header's frame count shows it."""
    full = make_noisy_avi(tmp_path / "full.avi", 30, rate=10)
    assert len(probe(full).pts) == 30
    data = full.read_bytes()
    cut = tmp_path / "cut.avi"
    cut.write_bytes(data[: len(data) // 2])
    with av.open(str(cut)) as container:
        assert sum(1 for _ in container.decode(video=0)) < 30, "precondition: the cut file must decode short"

    with pytest.raises(ValueError, match="header says 30 frames"):
        probe(cut)


def test_probe_image_is_one_frame(tmp_path: Path) -> None:
    info = probe(make_image(tmp_path / "x.jpg"))
    assert (info.kind, info.width, info.height, info.pts.tolist()) == ("image", 48, 32, [0.0])


def test_probe_refuses_unknown_suffix(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hi")
    with pytest.raises(ValueError, match="Not a video or image"):
        probe(tmp_path / "notes.txt")


def test_probe_refuses_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")
