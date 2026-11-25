"""The Master: a handle on the store. It finds runs and answers queries; it does no model work."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import duckdb

from rin.run import Run
from rin.store import Layout, RunWriter
from rin.store.views import refresh


class Master:
    """Open (or create) a store:

        m = Master("workspace/store")
        m.sql("SELECT split, count(*) FROM media GROUP BY split").df()
    """

    def __init__(self, root: str | Path) -> None:
        self.layout = Layout(Path(root).resolve())
        self.layout.create()
        leftovers = sorted(p.name for p in self.layout.tmp.iterdir())
        if leftovers:
            # A run killed hard (out of memory, WSL2 crash) never cleans up; runs/ is still
            # complete, but these folders can be gigabytes.
            warnings.warn(
                f"{len(leftovers)} unfinished run folder(s) in {self.layout.tmp}: {leftovers[:5]}. "
                "If no run is in progress, delete them.",
                stacklevel=2,
            )
        self._con = duckdb.connect()  # in memory: the views point at the files, nothing is copied

    def __repr__(self) -> str:
        return f"Master({str(self.layout.root)!r})"

    def sql(self, query: str, params: list[Any] | None = None) -> duckdb.DuckDBPyRelation:
        """Run SQL over the store's tables (runs, media, frames, frame_sets, ...).

        Returns a DuckDB relation: print it, or call .df() for pandas, .fetchall() for tuples.
        """
        refresh(self._con, self.layout)
        return self._con.sql(query, params=params)

    def run(self, run_id: str) -> Run:
        return Run(self, run_id)

    def runs(self) -> duckdb.DuckDBPyRelation:
        return self.sql("SELECT run_id, step, config, tags, inputs, who, finished_at FROM runs ORDER BY run_id")

    def writer(self, **kwargs: Any) -> RunWriter:
        """A RunWriter on this store. Steps use it; notebooks don't need it."""
        return RunWriter(self.layout, **kwargs)
