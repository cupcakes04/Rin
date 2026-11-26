"""Writing one run: into a temporary folder, then renamed into runs/ in one step (D-23).

A run that fails part-way leaves nothing in runs/, so every run you can see is complete.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets
import shutil
import subprocess
from collections.abc import Callable
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


def code_version() -> tuple[dict[str, Any], str]:
    """The Rin git commit the run was made with, and the uncommitted changes to src/ and the core
    environment (pyproject.toml, uv.lock) on top of it, as a patch ("" when they are clean). The run keeps the patch, so a run made from
    uncommitted code can still be traced to the code that made it."""
    here = Path(__file__).resolve().parent

    def git(*args: str, ok: tuple[int, ...] = (0,)) -> str:
        result = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True)
        if result.returncode not in ok:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    top = Path(git("rev-parse", "--show-toplevel").strip())
    src = str(top / "src")
    patch = git("diff", "--binary", "HEAD", "--", src, str(top / "pyproject.toml"), str(top / "uv.lock"))
    # git diff leaves out files git doesn't track yet; diffing each against /dev/null adds them
    # (exit code 1 means "they differ", which is the point).
    for name in git("ls-files", "--others", "--exclude-standard", "--", src).splitlines():
        patch += git("diff", "--binary", "--no-index", "--", "/dev/null", str(top / name), ok=(0, 1))
    info: dict[str, Any] = {"rin": git("rev-parse", "HEAD").strip(), "src_dirty": bool(patch)}
    if patch:
        info["src_diff_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
    return info, patch


def vendor_commit(repo: Path) -> str:
    """The git commit of a vendor engine's checkout (a submodule in vendors/), with "+dirty"
    when it has uncommitted changes: the engine's code is part of what made a run."""
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} in {repo} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    return commit + "+dirty" if git("status", "--porcelain") else commit


def _fsync(path: Path) -> None:
    """Push a file's or folder's contents from the OS's memory to the disk. Without it a hard
    crash (WSL2 has them) can leave a renamed file at 0 bytes: the rename reached the disk, the
    data didn't."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_durably(target: Path, write: Callable[[Path], None]) -> None:
    """Write a file under a temporary name, push it to disk, then rename it into place: after a
    crash the file is either whole or absent, never empty or half-written."""
    partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
    try:
        write(partial)
        _fsync(partial)
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    _fsync(target.parent)  # the rename itself reaches the disk


class RunWriter:
    """Use as a context manager:

        with RunWriter(layout, step="core.ingest", config={...}, inputs=[]) as w:
            w.write("media", table)
        w.run_id   # now in runs/

    If the block raises, the temporary folder is deleted and the error goes on up.
    `started_at` is when the writer is made: open it before the step's work, so the run's times
    cover that work.
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
        code, self.src_diff = code_version()
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
            "code": code,
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

    def add_provenance(
        self, *, checkpoint: str | None = None, checkpoint_sha256: str | None = None, **code: str
    ) -> None:
        """What is learned about a run while it runs: the exact checkpoint (revision, or the
        hash of its file), and the versions of the code it used (libraries in the job's
        environment, vendor engines' commits)."""
        if checkpoint is not None:
            self.info["checkpoint"] = checkpoint
        if checkpoint_sha256 is not None:
            self.info["checkpoint_sha256"] = checkpoint_sha256
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
        if self.src_diff:
            (self.tmp_dir / "src.diff").write_text(self.src_diff)
        self.info["finished_at"] = datetime.now().isoformat()
        (self.tmp_dir / "run.json").write_text(json.dumps(self.info, indent=2))
        # Every file and the folder itself reach the disk before the rename makes the run
        # visible, and the rename reaches the disk before the step returns.
        # Deepest first, so each folder is synced after what it holds (steps may write subfolders).
        for path in sorted(self.tmp_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            _fsync(path)
        _fsync(self.tmp_dir)
        os.rename(self.tmp_dir, self.layout.run_dir(self.run_id))
        _fsync(self.layout.runs)
