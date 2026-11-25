# Installing Rin

Rin runs on Linux (developed on WSL2 with one 16 GB GPU). This page sets up the **core environment**: the store, ingest and sampling. Each model (SAM3, DINOv3, YOLO, …) gets its own separate environment, added to this page when its servant is built.

## What you need first

| Tool | Why | Check |
|---|---|---|
| git | Rin and the vendor engines are git repos | `git --version` |
| [uv](https://docs.astral.sh/uv/) | Creates and fills every Rin environment | `uv --version` |
| curl | Only for downloading the MEVA dataset | `curl --version` |

uv downloads the right Python (3.12) by itself. Video decoding comes from PyAV, which ships its own FFmpeg libraries, so the core needs no system FFmpeg.

## 1. Get the code

```bash
git clone --recurse-submodules <repo-url> Rin
cd Rin
```

Already cloned without `--recurse-submodules`? Run `git submodule update --init` to fetch `vendors/`.

## 2. Create the core environment

```bash
uv sync
```

This reads `pyproject.toml` (what Rin needs) and `uv.lock` (the exact versions last tested), creates `.venv/`, and installs Rin itself in editable mode: a change in `src/` takes effect without reinstalling. Run it again after pulling changes; it only installs what changed.

Run anything inside the environment with `uv run`, e.g. `uv run python` or `uv run jupyter lab`.

## 3. Check it works

```bash
uv run pytest
```

Pass: every test passes, in a few seconds. The tests make their own tiny videos, so they need no data or GPU.

## 4. Get the data

```bash
bash dev/scripts/pull_meva.sh
```

Downloads the MEVA slice (about 5.3 GB) into `workspace/media/meva/`. Files already on disk are skipped. What the slice is and why: [dev/notes/heavy_data.md](../dev/notes/heavy_data.md).

## 5. First use

```python
from datetime import datetime
from rin import Master, core

def meva_name(path):            # MEVA keeps camera and start time in the filename
    day, start, _end, site, camera, _rest = path.name.split(".", 5)
    return {"site": site, "camera": camera,
            "started_at": datetime.strptime(f"{day} {start}", "%Y-%m-%d %H-%M-%S")}

m = Master("workspace/store")
pool = core.ingest(m, "workspace/media/meva/pool", split="pool", describe=meva_name)
frames = core.sample(m, pool, fps=2)

m.sql("SELECT camera, count(*) AS clips, sum(n_frames) AS frames FROM media GROUP BY camera").df()
```

Ingesting the 24 pool clips decodes every frame once, which took about 100 s on this machine. Everything Rin writes goes to `workspace/store/`; deleting that folder starts over. What each call does, and the rules behind it: [core.md](core.md).
