"""Writing one run: into a temporary folder, then renamed into runs/ in one step (D-23).

A run that fails part-way leaves nothing in runs/, so every run you can see is complete.
"""

from __future__ import annotations

import getpass
import json
import os
import secrets
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rin.store.layout import Layout
from rin.store.schema import conform, stored_schema


def new_run_id(step: str, now: datetime) -> str:
    """Sortable by time, readable, and no counter to keep in sync: time, step, 4 random characters."""
    return f"{now:%Y%m%d-%H%M%S}-{step}-{secrets.token_hex(2)}"


def code_version() -> dict[str, Any]:
    """The Rin git commit the run was made with, and whether src/ had uncommitted changes."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    top = git("rev-parse", "--show-toplevel")
    return {
        "rin": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", f"{top}/src")),
    }


class RunWriter:
    """Use as a context manager:

        with RunWriter(layout, step="core.ingest", config={...}, inputs=[]) as w:
            w.write("media", table)
        w.run_id   # now in runs/

    If the block raises, the temporary folder is deleted and the error goes on up.
    """

    def __init__(
        self,
        layout: Layout,
        *,
        step: str,
        config: dict[str, Any],
        inputs: list[str],
        tags: dict[str, Any] | None = None,
        model: str | None = None,
        checkpoint: str | None = None,
        checkpoint_sha256: str | None = None,
        who: str | None = None,
    ) -> None:
        """who: the person behind the run. Left out, it is the login user, which is right for
        any step the user starts; a step carrying someone else's work (their clicks, their
        drawn boxes) passes that person."""
        if who is not None and (not isinstance(who, str) or not who.strip()):
            raise ValueError(f"who must be a non-empty string, got {who!r}")
        self.layout = layout
        self.started_at = datetime.now()
        self.run_id = new_run_id(step, self.started_at)
        self.info: dict[str, Any] = {
            "run_id": self.run_id,
            "step": step,
            "model": model,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "config": config,
            "inputs": list(inputs),
            "tags": tags or {},
            "code": code_version(),
            "who": getpass.getuser() if who is None else who,
            "started_at": self.started_at.isoformat(),
            "finished_at": None,
        }
        json.dumps(self.info)  # fail now, not at the end, if config or tags aren't plain JSON
        self.tmp_dir = layout.tmp / self.run_id
        self.written: list[str] = []

    def __enter__(self) -> RunWriter:
        if self.layout.run_dir(self.run_id).exists():
            raise FileExistsError(f"Run {self.run_id} already exists")
        self.tmp_dir.mkdir(parents=True)
        return self

    def write(self, name: str, table: pa.Table) -> None:
        if name in self.written:
            raise ValueError(f"Run {self.run_id} already wrote table {name!r}")
        table = conform(name, table)
        table = table.add_column(0, "run_id", pa.array([self.run_id] * len(table), pa.string()))
        pq.write_table(table.cast(stored_schema(name)), self.tmp_dir / f"{name}.parquet")
        self.written.append(name)

    def add_provenance(self, *, checkpoint: str | None = None, **code: str) -> None:
        """What a job reports about itself after it ran: the exact checkpoint revision, and the
        versions of the libraries it used (they live in the job's environment, not this one)."""
        if checkpoint is not None:
            self.info["checkpoint"] = checkpoint
        self.info["code"].update(code)
        json.dumps(self.info)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            return
        if not self.written:
            shutil.rmtree(self.tmp_dir)
            raise RuntimeError(f"Run {self.run_id} wrote no tables; nothing was saved")
        self.info["finished_at"] = datetime.now().isoformat()
        (self.tmp_dir / "run.json").write_text(json.dumps(self.info, indent=2))
        os.rename(self.tmp_dir, self.layout.run_dir(self.run_id))
