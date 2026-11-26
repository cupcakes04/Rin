"""Starting a servant's job (D-07): `uv run` in the servant's folder, so the job gets that
folder's own environment, and its output streamed to the notebook as it comes.

Shared plumbing, like the frame cache: it knows nothing about any model's inputs or outputs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_job(job_dir: Path, spec_file: Path) -> None:
    """Run `job_dir/job.py spec_file` in `job_dir`'s environment and wait. Raises if it fails.

    `--locked` refuses to run if uv.lock is out of date with pyproject.toml, so the job never
    silently runs with packages other than the ones locked.
    """
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # the core env's, not the job's
    command = ["uv", "run", "--locked", "--project", str(job_dir), "python", str(job_dir / "job.py"), str(spec_file)]
    # Lines are printed as they come (a notebook's stdout is not a file a child process can share).
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as job:
        assert job.stdout is not None
        for line in job.stdout:
            print(line, end="", file=sys.stdout, flush=True)
    if job.returncode != 0:
        raise RuntimeError(f"The {job_dir.name} job failed (exit code {job.returncode}); its output is above")
