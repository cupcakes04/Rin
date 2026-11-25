"""Run: the small handle every step returns. The data stays on disk; this only knows the id."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
from jaxtyping import Float

from rin.store.schema import TABLES

if TYPE_CHECKING:
    from rin.master import Master


class Run:
    def __init__(self, master: Master, run_id: str) -> None:
        folder = master.layout.run_dir(run_id)
        if not (folder / "run.json").is_file():
            raise KeyError(f"No run {run_id!r} in {master.layout.runs}")
        self._m = master
        self.info: dict[str, Any] = json.loads((folder / "run.json").read_text())
        self.tables: list[str] = sorted(p.stem for p in folder.glob("*.parquet"))

    id = property(lambda self: self.info["run_id"])
    step = property(lambda self: self.info["step"])
    config = property(lambda self: self.info["config"])
    inputs = property(lambda self: self.info["inputs"])
    tags = property(lambda self: self.info["tags"])

    def __repr__(self) -> str:
        return f"Run({self.id!r}, tables={self.tables})"

    def table(self, name: str) -> duckdb.DuckDBPyRelation:
        """This run's rows of one table."""
        if name not in TABLES:
            raise KeyError(f"No table {name!r}. Known tables: {sorted(TABLES)}")
        if name not in self.tables:
            raise KeyError(f"Run {self.id} wrote no {name!r} table (it wrote {self.tables})")
        return self._m.sql(f"SELECT * EXCLUDE (run_id) FROM {name} WHERE run_id = ?", [self.id])

    def media(self) -> duckdb.DuckDBPyRelation:
        return self.table("media")

    def frames(self) -> duckdb.DuckDBPyRelation:
        return self.table("frames")

    def frame_set(self) -> duckdb.DuckDBPyRelation:
        return self.table("frame_sets")

    def frame_scores(self) -> duckdb.DuckDBPyRelation:
        return self.table("frame_scores")

    def embeddings(self) -> duckdb.DuckDBPyRelation:
        """Which frame (and kind) each row of `vectors()` belongs to."""
        return self.table("embeddings")

    def vectors(self) -> Float[np.ndarray, "rows dim"]:
        """The run's embeddings.npy, read lazily from disk. Row r belongs to the `embeddings`
        row whose `row` is r."""
        if "embeddings" not in self.tables:
            raise TypeError(f"Run {self.id} wrote no embeddings (it wrote {self.tables})")
        return np.load(self._m.layout.run_dir(self.id) / "embeddings.npy", mmap_mode="r")

    def __len__(self) -> int:
        """Number of frames, for a run that wrote a frame set."""
        if "frame_sets" not in self.tables:
            raise TypeError(f"Run {self.id} wrote no frame set, so it has no length (it wrote {self.tables})")
        (n,) = self._m.sql("SELECT count(*) FROM frame_sets WHERE run_id = ?", [self.id]).fetchall()[0]
        return n
