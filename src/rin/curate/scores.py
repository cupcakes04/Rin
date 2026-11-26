"""curate.score_frames and curate.keep: measure cheap numbers about each frame once, then
choose frames with thresholds over the stored numbers (D-26).
"""

from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
from jaxtyping import Float

from rin.master import Master
from rin.media import Rgb, read_frames
from rin.run import Run

SCORE_WIDTH = 320
"""Frames are scored on a copy this wide (height keeps the aspect ratio): 1920×1080 -> 320×180.
The scores only need coarse structure, and the small copy keeps scoring cheap."""

DARK_BELOW = 16 / 255
BRIGHT_ABOVE = 239 / 255

SCORES = {
    "motion": "Mean absolute brightness change from the previous sampled frame of the same media "
              "(0-1). The first sampled frame is compared with the next one. Not set when a media "
              "has only one frame in the set (an image)",
    "blur": "Variance of the Laplacian of the brightness (0-1 scale): low means few sharp edges",
    "dark": f"Share of pixels darker than {DARK_BELOW * 255:.0f}/255",
    "bright": f"Share of pixels brighter than {BRIGHT_ABOVE * 255:.0f}/255",
}
"""What each score means. The names are what `keep`'s `where` text refers to."""

Gray = Float[np.ndarray, "height width"]


def _gray(rgb: Rgb) -> Gray:
    return rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0


def _blur(gray: Gray) -> float:
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def score_frames(m: Master, frames: Run, names: list[str]) -> Run:
    """Measure each frame of a frame set. Decodes video (CPU); the numbers are stored so that
    choosing thresholds later (`keep`) never decodes again. Names: see SCORES."""
    unknown = [n for n in names if n not in SCORES]
    if not names or unknown or len(set(names)) != len(names):
        raise ValueError(f"names must be distinct, from {sorted(SCORES)}; got {names!r}")
    if "frame_sets" not in frames.tables:
        raise TypeError(f"score_frames needs a frame set; {frames.id} wrote {frames.tables}")

    config = {"names": names, "width": SCORE_WIDTH, "dark_below": DARK_BELOW, "bright_above": BRIGHT_ABOVE}
    with m.writer(step="curate.score_frames", config=config, inputs=[frames.id]) as w:
        per_media = m.sql(
            """SELECT d.media_id, d.path, d.width, d.height, list(s.frame_idx ORDER BY s.frame_idx)
               FROM frame_sets s JOIN media d USING (media_id)
               WHERE s.run_id = ? GROUP BY ALL ORDER BY d.path""",
            [frames.id],
        ).fetchall()

        cols: dict[str, list] = {"media_id": [], "frame_idx": [], "name": [], "value": []}

        def add(media_id: str, idx: list[int], name: str, values: list[float]) -> None:
            cols["media_id"] += [media_id] * len(idx)
            cols["frame_idx"] += idx
            cols["name"] += [name] * len(idx)
            cols["value"] += values

        for n, (media_id, path, width, height, idx) in enumerate(per_media, start=1):
            started = time.perf_counter()
            size = (SCORE_WIDTH, max(2, round(SCORE_WIDTH * height / width / 2) * 2))
            grays = [_gray(rgb) for _, rgb in read_frames(m.layout.media_file(path), idx, size=size)]
            if "motion" in names and len(grays) > 1:
                steps = [float(np.abs(b - a).mean()) for a, b in zip(grays, grays[1:], strict=False)]
                add(media_id, idx, "motion", [steps[0], *steps])
            if "blur" in names:
                add(media_id, idx, "blur", [_blur(g) for g in grays])
            if "dark" in names:
                add(media_id, idx, "dark", [float((g < DARK_BELOW).mean()) for g in grays])
            if "bright" in names:
                add(media_id, idx, "bright", [float((g > BRIGHT_ABOVE).mean()) for g in grays])
            print(f"[{n}/{len(per_media)}] {path}: {len(idx)} frames, {time.perf_counter() - started:.1f} s")

        w.write("frame_scores", pa.table(cols))
    return m.run(w.run_id)


def keep(m: Master, scores: Run, *, where: str) -> Run:
    """The frames of a scoring run whose scores pass `where`, an SQL condition over the score
    names, e.g. "motion >= 0.01 AND blur >= 0.002". Saved in the run's config, so the
    threshold is never lost. A frame missing a score named in `where` (an image has no motion)
    does not pass. Choosing new thresholds is a new keep run; nothing is decoded again.
    """
    if not isinstance(where, str) or not where.strip():
        raise ValueError(f"where must be an SQL condition, got {where!r}")
    if "frame_scores" not in scores.tables:
        raise TypeError(f"keep needs a score_frames run; {scores.id} wrote {scores.tables}")
    names = [n for (n,) in m.sql(
        "SELECT DISTINCT name FROM frame_scores WHERE run_id = ? ORDER BY name", [scores.id]
    ).fetchall()]
    if any(n not in SCORES for n in names):  # names become column names below
        raise ValueError(f"{scores.id} holds scores Rin doesn't know: {names}")
    columns = ", ".join(f"max(value) FILTER (WHERE name = '{n}') AS {n}" for n in names)
    passed = m.sql(
        f"""SELECT media_id, frame_idx FROM (
                SELECT media_id, frame_idx, {columns}
                FROM frame_scores WHERE run_id = ? GROUP BY media_id, frame_idx)
            WHERE {where} ORDER BY media_id, frame_idx""",
        [scores.id],
    ).fetchall()
    if not passed:
        raise ValueError(f"No frame of {scores.id} passes {where!r}")
    table = pa.table({
        "media_id": [media_id for media_id, _ in passed],
        "frame_idx": pa.array([idx for _, idx in passed], pa.int64()),
        "rank": pa.nulls(len(passed), pa.int64()),
    })
    with m.writer(step="curate.keep", config={"where": where}, inputs=[scores.id]) as w:
        w.write("frame_sets", table)
    return m.run(w.run_id)
