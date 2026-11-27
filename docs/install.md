# Installing Rin

Rin runs on Linux (developed on WSL2 with one 16 GB GPU). This page sets up the **core environment** (the store, ingest, sampling, curation) and, in sections 6 to 9, the DINOv3 and SAM3 environments, the viewer (FiftyOne) and the YOLO environment. Each model gets its own separate environment, added to this page when its servant is built.

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

Ingesting the 24 pool clips decodes every frame once, which took about 100 s on this machine. Everything Rin writes goes to `workspace/store/`. Once it holds a person's work (a `human.annotate` run), never delete it to start over: only its frame cache, `store/frames/`, is safe to delete. Back up the human runs after each labelling session: `bash dev/scripts/backup_human_runs.sh` writes a dated archive to `workspace/backups/` (it survives a deleted store, not a broken disk; pass a folder to put it elsewhere). What each call does, and the rules behind it: [core.md](core.md).

## 6. The DINOv3 environment (GPU)

`dinov3.embed` runs its model in a separate environment, defined in `src/rin/servants/dinov3/pyproject.toml` (PyTorch built for CUDA 13.0, `transformers`). The first `embed` call creates it; to create it ahead of time:

```bash
uv sync --locked --project src/rin/servants/dinov3
```

Check: `src/rin/servants/dinov3/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"` prints `True`. If it prints `False`, the NVIDIA driver doesn't support CUDA 13.0; `nvidia-smi` shows the version it supports.

The weights load from the Hugging Face cache. DINOv3 is gated: accept its licence on the [model page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), run `hf auth login` once, then `hf download facebook/dinov3-vitl16-pretrain-lvd1689m`.

## 7. The SAM3 environment (GPU)

`sam3.concept` runs SAM3 in a separate environment, defined in `src/rin/servants/sam3/pyproject.toml`: the user's engine `vendors/sam3_engine` (installed editable, so it runs the submodule's code), PyTorch for CUDA 13.0, and `numpy<2`, which the engine needs and the core can't share. Fetch the engine, then create the environment (the first `concept` call also creates it):

```bash
git submodule update --init vendors/sam3_engine
uv sync --locked --project src/rin/servants/sam3
```

Check: `src/rin/servants/sam3/.venv/bin/python -c "import torch, sam3_engine; print(torch.cuda.is_available())"` prints `True`.

The checkpoint is not downloaded by Rin: put `sam3.pt` from [facebook/sam3](https://huggingface.co/facebook/sam3) (gated: accept the licence first) in `workspace/checkpoints/sam3/`, and pass its path to `concept`.

## 8. The viewer environment (FiftyOne)

`viewer.show` and `human.annotate` run FiftyOne in a separate environment, defined in `src/rin/viewer/pyproject.toml`. FiftyOne keeps its datasets in MongoDB, and on Ubuntu 26.04 its `fiftyone-db` package installs without MongoDB's `mongod`. One script creates the environment and adds `mongod` (MongoDB's official Ubuntu 24.04 build, checked against its published SHA-256; no sudo):

```bash
bash dev/scripts/fiftyone_mongod.sh
```

Check: the script's last line is `db version v8.0.17`. If it says a library is missing, install that library with `apt` and run the script again. If FiftyOne later says ``Could not find `mongod` ``, run the script again: a fresh `uv sync` that reinstalls `fiftyone-db` removes the binary.

Look and draw in the browser: `uv run --project src/rin/viewer fiftyone app launch <name>`, then open `http://localhost:5151`.

## 9. The YOLO environment (GPU)

`yolo.train` runs Ultralytics in a separate environment, defined in `src/rin/servants/yolo/pyproject.toml` (Ultralytics, PyTorch for CUDA 13.0). The first `train` call creates it; to create it ahead of time:

```bash
uv sync --locked --project src/rin/servants/yolo
```

Check: `src/rin/servants/yolo/.venv/bin/python -c "import torch, ultralytics; print(torch.cuda.is_available())"` prints `True`.

Stock checkpoints live in `workspace/checkpoints/yolo/` (e.g. `yolo26s.pt`); pass the file to `train`. The job runs in that folder, where Ultralytics' mixed-precision check looks for `yolo26n.pt`: keep `yolo26n.pt` there, or Ultralytics downloads it into the folder on the first run.
