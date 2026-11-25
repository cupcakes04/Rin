"""Shared fixtures: a throwaway workspace with a store, and tiny generated videos.

Every test video is made here at test time, so no binary files live in git. Frame i of a
clip is solid gray at level GRAY_STEP * i, so a decoded pixel tells which frame came back.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from rin import Master

GRAY_STEP = 8
SIZE = (64, 64)  # width, height

# Declares 30 fps, but only the first 12 frames follow it; the next 20 are 0.25 s apart.
VFR_STARTS_MS = [round(i * 1000 / 30) for i in range(12)] + [400 + 250 * k for k in range(20)]
# 16 frames at a true 10 fps.
CFR_STARTS_MS = [100 * i for i in range(16)]


def make_clip(path: Path, starts_ms: list[int], *, rate: int) -> Path:
    """Lossless H.264 with frames starting at the given milliseconds (MKV keeps real timestamps)."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = SIZE
        stream.pix_fmt = "yuv444p"
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {"qp": "0"}
        for i, start in enumerate(starts_ms):
            pixels = np.full((SIZE[1], SIZE[0], 3), GRAY_STEP * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = start
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_avi_with_bframes(path: Path, n_frames: int, *, rate: int) -> Path:
    """An AVI like MEVA's: H.264 with B-frames, so frames are stored out of display order and
    the demuxer attaches timestamps in storage order. Near-lossless (crf 1), because lossless
    x264 turns B-frames off; gray levels 8 apart still identify each frame."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = SIZE
        stream.pix_fmt = "yuv444p"
        stream.codec_context.time_base = Fraction(1, rate)
        stream.options = {"crf": "1", "bf": "2", "x264-params": "b-adapt=0"}
        for i in range(n_frames):
            pixels = np.full((SIZE[1], SIZE[0], 3), GRAY_STEP * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_noisy_avi(path: Path, n_frames: int, *, rate: int) -> Path:
    """An AVI of random-noise frames. Noise doesn't compress, so the frames fill most of the
    file and cutting it short loses frames without breaking the header (as with a real clip)."""
    rng = np.random.default_rng(0)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            pixels = rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_image(path: Path) -> Path:
    with av.open(str(path), mode="w", format="image2") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width, stream.height = 48, 32
        stream.pix_fmt = "yuvj420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 48, 3), np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def decoded_gray_ids(path: Path) -> list[int]:
    """Which generated frame each decoded frame is, in the order the decoder returns them."""
    with av.open(str(path)) as container:
        return [
            round(float(f.to_ndarray(format="rgb24")[0, 0, 0]) / GRAY_STEP)
            for f in container.decode(video=0)
        ]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "workspace" / "media").mkdir(parents=True)
    return tmp_path / "workspace"


@pytest.fixture
def m(workspace: Path) -> Master:
    return Master(workspace / "store")


@pytest.fixture
def clips(workspace: Path) -> Path:
    """A media folder with a VFR clip, a CFR clip and an image."""
    folder = workspace / "media" / "clips"
    folder.mkdir()
    make_clip(folder / "vfr.mkv", VFR_STARTS_MS, rate=30)
    make_clip(folder / "cfr.mkv", CFR_STARTS_MS, rate=10)
    make_image(folder / "still.jpg")
    return folder
