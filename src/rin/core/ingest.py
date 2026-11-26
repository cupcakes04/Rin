"""core.ingest: media files -> one media row per file, one frames row per decoded frame."""

from __future__ import annotations

import hashlib
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


def ingest(
    m: Master, source: str | Path | list[str | Path], *, split: str, describe: Describe | None = None
) -> Run:
    """Add a file, a list of files, or every video and image directly inside a folder, to the store.

    split:    "pool", "test", "sample", ... Set per file, so a recording is never split by frame.
    describe: optional function path -> {"site", "camera", "started_at"}; facts Rin can't read
              from the file itself (e.g. MEVA keeps them in the filename). Without it they are null.

    Refuses files already ingested (by path, or the same content under another path), and
    files outside the workspace. A media's id comes from its content, so re-ingesting the same
    file into a fresh store gives it the same id.
    """
    if not isinstance(split, str) or not split:
        raise ValueError(f"split must be a non-empty string, got {split!r}")
    paths = _collect(source)
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
        "source": (
            [m.layout.relative_media_path(Path(s)) for s in source] if isinstance(source, list)
            else m.layout.relative_media_path(Path(source))
        ),
        "split": split,
        "describe": None if describe is None else getattr(describe, "__name__", repr(describe)),
    }
    with m.writer(step="core.ingest", config=config, inputs=[]) as w:
        media_ids = _content_ids(m, paths, rel_paths)  # reads every file once: part of the run's time
        media_rows: list[dict[str, Any]] = []
        frame_tables: list[pa.Table] = []
        for i, (path, rel, extra, media_id) in enumerate(zip(paths, rel_paths, extras, media_ids, strict=True), start=1):
            info = probe(path)
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


def _collect(source: str | Path | list[str | Path]) -> list[Path]:
    if isinstance(source, list):
        paths = [Path(s) for s in source]
        missing = [str(p) for p in paths if not p.is_file()]
        if not paths or missing:
            raise FileNotFoundError(f"Not files: {missing[:5]}" if missing else "The list of files is empty")
        if len(set(p.resolve() for p in paths)) != len(paths):
            raise ValueError("A file is listed twice")
        return paths
    source = Path(source)
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


def _content_ids(m: Master, paths: list[Path], rel_paths: list[str]) -> list[str]:
    """Each file's media_id, made from its bytes: the same file ingested again gets the same id,
    so judgments a person made against it still point at it. The same bytes under two paths
    are refused (two copies would be two "different" recordings, both in a split)."""
    ids = [_content_id(p) for p in paths]
    first: dict[str, str] = {}
    for media_id, rel in zip(ids, rel_paths, strict=True):
        if media_id in first:
            raise ValueError(f"{rel} and {first[media_id]} are the same file (same content); ingest one")
        first[media_id] = rel
    stored = m.sql("SELECT media_id, path FROM media WHERE list_contains(?, media_id)", [ids]).fetchall()
    if stored:
        pairs = [f"{first[media_id]} (already ingested as {path})" for media_id, path in sorted(stored)]
        raise ValueError(f"Same content already in the store: {pairs[:5]}")
    return ids


def _content_id(path: Path) -> str:
    """"m" + the first 16 hex characters (64 bits) of the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return "m" + digest.hexdigest()[:16]
