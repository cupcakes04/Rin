"""DINOv3 embedding job. Runs in this folder's own environment; imports nothing from Rin.

    uv run --project src/rin/servants/dinov3 python job.py <spec.json>

Reads spec.json (model, size, what, batch_size, images) and writes, next to it:
- embeddings.npy: float16, (len(what) * len(images), dim). Rows are grouped by kind in the
  order of `what`; within a kind, in the order of `images`.
- job.json: the model revision and library versions, for the run's provenance.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

PREFETCH = 3
"""Batches read and preprocessed ahead on CPU threads while the GPU works."""


def main(spec_file: Path) -> None:
    spec = json.loads(spec_file.read_text())
    out_dir = spec_file.parent
    width, height = spec["size"]
    what, paths, batch_size = spec["what"], spec["images"], spec["batch_size"]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to the DINOv3 job")

    processor = AutoImageProcessor.from_pretrained(spec["model"])
    # float32 weights + float16 autocast: matrix products run in float16, while the steps that
    # can overflow stay in float32. Weights loaded *as* float16 overflow inside the network and
    # every vector comes out NaN (measured here); this setting matches float32 (cosine 1.0000).
    model = AutoModel.from_pretrained(spec["model"], dtype=torch.float32, attn_implementation="sdpa")
    model = model.cuda().eval()
    registers = model.config.num_register_tokens
    n = len(paths)
    vectors = np.lib.format.open_memmap(
        out_dir / "embeddings.npy", mode="w+", dtype=np.float16, shape=(len(what) * n, model.config.hidden_size)
    )

    def prepare(batch: list[str]) -> torch.Tensor:
        images = []
        for p in batch:
            with Image.open(p) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, size={"height": height, "width": width}, return_tensors="pt")
        return inputs["pixel_values"]

    starts = range(0, n, batch_size)
    started = time.perf_counter()
    autocast = torch.autocast("cuda", dtype=torch.float16)
    with ThreadPoolExecutor(max_workers=PREFETCH) as pool, torch.inference_mode(), autocast:
        pending: list[Future[torch.Tensor]] = [pool.submit(prepare, paths[s : s + batch_size]) for s in starts[:PREFETCH]]
        for b, s in enumerate(starts):
            pixels = pending.pop(0).result()
            if b + PREFETCH < len(starts):
                nxt = starts[b + PREFETCH]
                pending.append(pool.submit(prepare, paths[nxt : nxt + batch_size]))
            if pixels.shape[-2:] != (height, width):
                raise RuntimeError(f"The processor made {tuple(pixels.shape[-2:])}, not {(height, width)}")
            tokens = model(pixel_values=pixels.to("cuda")).last_hidden_state.float()
            if not torch.isfinite(tokens).all():
                raise RuntimeError(f"Non-finite DINOv3 output in the batch starting at image {s}")
            # tokens: [CLS, `registers` register tokens, patch tokens...]
            by_kind = {"cls": tokens[:, 0], "mean_patch": tokens[:, 1 + registers :].mean(dim=1)}
            for k, kind in enumerate(what):
                vectors[k * n + s : k * n + s + len(pixels)] = by_kind[kind].to(torch.float16).cpu().numpy()
            if (b + 1) % 20 == 0 or b + 1 == len(starts):
                done = min(s + batch_size, n)
                print(f"{done}/{n} frames, {done / (time.perf_counter() - started):.1f} frames/s", flush=True)
    vectors.flush()
    del vectors

    (out_dir / "job.json").write_text(json.dumps({
        "model_revision": _revision(spec["model"]),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }))


def _revision(model: str) -> str | None:
    """The Hugging Face commit the weights came from: the cache keeps each download under
    snapshots/<commit>/. None when `model` is a local folder, which has no commit."""
    if Path(model).is_dir():
        return None
    config = try_to_load_from_cache(model, "config.json")
    if not isinstance(config, str):
        raise RuntimeError(f"{model} loaded, but its config.json isn't in the Hugging Face cache")
    return Path(config).parent.name


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: job.py <spec.json>")
    main(Path(sys.argv[1]))
