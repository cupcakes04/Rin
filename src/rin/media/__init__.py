"""Reading media files: sizes, timestamps, time -> frame, pixels. Knows nothing about the store."""

from rin.media.probe import IMAGE_SUFFIXES, VIDEO_SUFFIXES, Probe, media_kind, probe
from rin.media.reader import Rgb, read_frames
from rin.media.timing import TICK_TOLERANCE, frame_on_screen, sample_frames

__all__ = [
    "IMAGE_SUFFIXES",
    "TICK_TOLERANCE",
    "VIDEO_SUFFIXES",
    "Probe",
    "Rgb",
    "frame_on_screen",
    "media_kind",
    "probe",
    "read_frames",
    "sample_frames",
]
