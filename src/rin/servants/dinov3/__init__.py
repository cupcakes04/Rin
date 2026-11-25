"""dinov3.embed: one DINOv3 vector per frame, computed by a batch job in the DINOv3 environment.

The caller (this file) runs in the core environment. The job (job.py) runs in this folder's own
environment and imports nothing from Rin. They meet in the run's temporary folder:

    caller: cache the frames, open the run, write spec.json  ->  job: write embeddings.npy, job.json
    caller: check the array, write the embeddings table, file the run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

from rin.servants.frame_cache import cached_frames
from rin.master import Master
from rin.run import Run

JOB_DIR = Path(__file__).resolve().parent
KINDS = ("cls", "mean_patch")
"""cls: the class token. mean_patch: the mean of the patch tokens, register tokens left out."""


def embed(
    m: Master,
    frames: Run,
    *,
    model: str,
    size: tuple[int, int] = (896, 512),
    what: list[str] | tuple[str, ...] = ("cls",),
    batch_size: int = 32,
) -> Run:
    """Embed every frame of a frame set with DINOv3 (GPU).

    model:      a Hugging Face DINOv3 id, e.g. "facebook/dinov3-vitl16-pretrain-lvd1689m".
    size:       (width, height) the frames are resized to; multiples of 16 (the patch size).
                The default keeps 16:9, so a person 100 px tall at 1080p spans about 3 patches.
    what:       vector kinds, from KINDS. One row per frame and kind in embeddings.npy.
    batch_size: frames per forward pass.
    """
    if not isinstance(model, str) or not model:
        raise ValueError(f"model must be a Hugging Face model id, got {model!r}")
    if len(size) != 2 or any(not isinstance(s, int) or isinstance(s, bool) or s < 16 or s % 16 for s in size):
        raise ValueError(f"size must be (width, height), each a multiple of 16, got {size!r}")
    what = list(what)
    if not what or any(k not in KINDS for k in what) or len(set(what)) != len(what):
        raise ValueError(f"what must be distinct kinds from {KINDS}, got {what!r}")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    images = cached_frames(m, frames)
    config = {"size": list(size), "what": what, "batch_size": batch_size}
    with m.writer(step="dinov3.embed", config=config, inputs=[frames.id], model=model) as w:
        spec = {"model": model, **config, "images": [str(p) for _, _, p in images]}
        spec_file = w.tmp_dir / "spec.json"
        spec_file.write_text(json.dumps(spec))
        _run_job(spec_file)

        report = json.loads((w.tmp_dir / "job.json").read_text())
        vectors = np.load(w.tmp_dir / "embeddings.npy", mmap_mode="r")
        n = len(images)
        if vectors.ndim != 2 or vectors.shape[0] != n * len(what) or vectors.dtype != np.float16:
            raise RuntimeError(
                f"The job wrote {vectors.shape} {vectors.dtype}; expected ({n * len(what)}, dim) float16"
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("The job wrote non-finite values (NaN or inf) into embeddings.npy")
        del vectors
        w.add_provenance(checkpoint=report["model_revision"], **report["versions"])
        w.write("embeddings", pa.table({
            "row": pa.array(range(n * len(what)), pa.int64()),
            "media_id": [media_id for media_id, _, _ in images] * len(what),
            "frame_idx": pa.array([idx for _, idx, _ in images] * len(what), pa.int64()),
            "instance_idx": pa.nulls(n * len(what), pa.int64()),
            "kind": [k for k in what for _ in range(n)],
        }))
        spec_file.unlink()  # absolute paths to cached frames: derived, not provenance
        (w.tmp_dir / "job.json").unlink()
    return m.run(w.run_id)


def _run_job(spec_file: Path) -> None:
    """Run job.py in this folder's environment and wait. Its output goes straight to the
    notebook. `--locked` refuses to run if uv.lock is out of date with pyproject.toml."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # the core env's, not the job's
    command = ["uv", "run", "--locked", "--project", str(JOB_DIR), "python", str(JOB_DIR / "job.py"), str(spec_file)]
    # Lines are printed as they come (a notebook's stdout is not a file a child process can share).
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as job:
        assert job.stdout is not None
        for line in job.stdout:
            print(line, end="", file=sys.stdout, flush=True)
    if job.returncode != 0:
        raise RuntimeError(f"The DINOv3 job failed (exit code {job.returncode}); its output is above")
