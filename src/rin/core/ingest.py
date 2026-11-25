"""core.ingest: media files -> one media row per file, one frames row per decoded frame."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from rin.master import Master
from rin.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES, media_kind, probe
from rin.run import Run

DESCRIBE_KEYS = {"site": str, "camera": str, "started_at": datetime}
"""What a `describe` function may return. Anything else is refused."""

Describe = Callable[[Path], dict[str, Any]]


def ingest(m: Master, source: str | Path, *, split: str, describe: Describe | None = None) -> Run:
    """Add a file, or every video and image directly inside a folder, to the store.

    split:    "pool", "test", "sample", ... Set per file, so a recording is never split by frame.
    describe: optional function path -> {"site", "camera", "started_at"}; facts Rin can't read
              from the file itself (e.g. MEVA keeps them in the filename). Without it they are null.

    Refuses files already ingested, and files outside the workspace.
    """
    if not isinstance(split, str) or not split:
        raise ValueError(f"split must be a non-empty string, got {split!r}")
    paths = _collect(Path(source))
    rel_paths = [m.layout.relative_media_path(p) for p in paths]
    # Checked for every file before any is decoded: a bad filename at clip 24 fails in
    # milliseconds, not after decoding clips 1-23.
    for p in paths:
        media_kind(p)  # raises on a suffix Rin doesn't read
    extras = [_described(describe, p) for p in paths]

    already = m.sql("SELECT path FROM media WHERE list_contains(?, path)", [rel_paths]).fetchall()
    if already:
        raise ValueError(f"Already ingested: {sorted(p for (p,) in already)[:5]} (and maybe more)")

    config = {
        "source": m.layout.relative_media_path(Path(source)),
        "split": split,
        "describe": None if describe is None else getattr(describe, "__name__", repr(describe)),
    }
    taken_ids: set[str] = {mid for (mid,) in m.sql("SELECT media_id FROM media").fetchall()}
    with m.writer(step="core.ingest", config=config, inputs=[]) as w:
        media_rows: list[dict[str, Any]] = []
        frame_tables: list[pa.Table] = []
        for i, (path, rel, extra) in enumerate(zip(paths, rel_paths, extras, strict=True), start=1):
            info = probe(path)
            media_id = _new_media_id(taken_ids)
            media_rows.append({
                "media_id": media_id, "path": rel, "kind": info.kind,
                "width": info.width, "height": info.height, "n_frames": len(info.pts),
                "site": extra.get("site"), "camera": extra.get("camera"),
                "started_at": extra.get("started_at"), "split": split,
            })
            frame_tables.append(pa.table({
                "media_id": [media_id] * len(info.pts),
                "frame_idx": pa.array(range(len(info.pts)), pa.int64()),
                "pts": info.pts,
            }))
            print(f"[{i}/{len(paths)}] {rel}: {len(info.pts)} frames")
        w.write("media", pa.Table.from_pylist(media_rows))
        w.write("frames", pa.concat_tables(frame_tables))
    return m.run(w.run_id)


def _collect(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    known = VIDEO_SUFFIXES | IMAGE_SUFFIXES
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in known)
    if not paths:
        raise ValueError(f"No video or image files directly inside {source}")
    return paths


def _described(describe: Describe | None, path: Path) -> dict[str, Any]:
    if describe is None:
        return {}
    extra = describe(path)
    if not isinstance(extra, dict):
        raise TypeError(f"describe({path.name}) must return a dict, got {type(extra).__name__}")
    unknown = set(extra) - set(DESCRIBE_KEYS)
    if unknown:
        raise ValueError(f"describe({path.name}) returned unknown keys {sorted(unknown)}; allowed: {sorted(DESCRIBE_KEYS)}")
    for key, value in extra.items():
        if value is not None and not isinstance(value, DESCRIBE_KEYS[key]):
            raise TypeError(f"describe({path.name})[{key!r}] must be {DESCRIBE_KEYS[key].__name__}, got {value!r}")
    started_at = extra.get("started_at")
    if started_at is not None and started_at.tzinfo is not None:
        # The column stores local wall-clock time with no zone; a zone would be silently converted.
        raise ValueError(
            f"describe({path.name})['started_at'] must be local time with no timezone, got {started_at!r}"
        )
    return extra


def _new_media_id(taken: set[str]) -> str:
    while True:
        media_id = "m" + secrets.token_hex(4)
        if media_id not in taken:
            taken.add(media_id)
            return media_id
