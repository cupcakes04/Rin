"""M4 step 6c: the pool and val chains with quiet footage (D-41, D-31), SAM3 on the new candidates,
and the nine selections.

    uv run python dev/research/m4_selection/pool_chain.py

Pool: 2018-03-07 09:55-10:55 (busy) + 17:10-17:40 (quiet); val: 11:30-11:40 + 17:00-17:10.
Chain per split: sample 2 fps -> DINOv3 CLS -> dedupe 0.99 (no motion filter: D-41).
SAM3 labels only candidates its M4 run (`5477`) didn't (D-37). Resumable: a run with the same
step, inputs and config is reused.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from rin import Master, core, curate
from rin.servants import dinov3, sam3

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REV = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
MEDIA = Path("workspace/media/meva")
M4_LABELLED = "20261127-111139-curate.union-365e"   # the frames SAM3 run 5477 labelled
M4_LABELS = "20261127-111139-sam3.concept-5477"

m = Master("workspace/store")


def meva_name(path):
    day, start, _end, site, camera, _rest = path.name.split(".", 5)
    return {"site": site, "camera": camera,
            "started_at": datetime.strptime(f"{day} {start}", "%Y-%m-%d %H-%M-%S")}


def existing(step: str, inputs: list[str], **config) -> str | None:
    """The latest run of `step` on exactly these inputs whose config holds these values."""
    for run_id, cfg in m.sql("SELECT run_id, config FROM runs WHERE step = ? AND inputs = ? ORDER BY run_id DESC",
                             [step, inputs]).fetchall():
        if all(json.loads(cfg).get(k) == v for k, v in config.items()):
            return run_id
    return None


def reuse(run_id, make):
    return m.run(run_id) if run_id else make()


def ingest_quiet(folder: str, pattern: str, split: str):
    files = sorted(MEDIA / folder / p.name for p in (MEDIA / folder).glob(pattern))
    rel = [f"media/meva/{folder}/{f.name}" for f in files]
    found = [r for r, c in m.sql("SELECT run_id, config FROM runs WHERE step = 'core.ingest'").fetchall()
             if json.loads(c)["source"] == rel]
    return reuse(found[0] if found else None, lambda: core.ingest(m, [str(f) for f in files], split=split, describe=meva_name))


ingests = {
    "pool": [m.run("20261126-113143-core.ingest-7de8"), ingest_quiet("pool", "2018-03-07.17-*", "pool")],
    "val": [m.run("20261127-104452-core.ingest-eea1"), ingest_quiet("val", "2018-03-07.17-*", "val")],
}
cands = {}
for name, runs in ingests.items():
    t = time.time()
    samples = [reuse(existing("core.sample", [r.id], fps=2), lambda r=r: core.sample(m, r, fps=2)) for r in runs]
    frames = reuse(existing("curate.union", [s.id for s in samples]), lambda: curate.union(m, samples))
    emb = reuse(existing("dinov3.embed", [frames.id]), lambda: dinov3.embed(m, frames, model=MODEL, revision=REV))
    cands[name] = reuse(existing("curate.dedupe", [frames.id, emb.id], cosine_max=0.99),
                        lambda: curate.dedupe(m, frames, emb, cosine_max=0.99))
    print(name, "frames", frames.id, len(frames), "emb", emb.id, "cands", cands[name].id, len(cands[name]),
          f"{time.time() - t:.0f} s", flush=True)
    if name == "pool":
        pool_emb = emb

both = reuse(existing("curate.union", [cands["pool"].id, cands["val"].id]), lambda: curate.union(m, [cands["pool"], cands["val"]]))
new = reuse(existing("curate.difference", [both.id, M4_LABELLED]), lambda: curate.difference(m, both, [m.run(M4_LABELLED)]))
print("candidates not labelled in M4:", new.id, len(new), flush=True)
t = time.time()
labels = reuse(existing("sam3.concept", [new.id]), lambda: sam3.concept(
    m, new, prompts={"person": "person"}, checkpoint="workspace/checkpoints/sam3/sam3.pt", score_floor=0.1, masks=True))
print("sam3", labels.id, f"{time.time() - t:.0f} s", "(with", M4_LABELS, "it labels every candidate)", flush=True)

for method, extra in (("random", {}), ("kcenter", {}), ("kmeans_stratified", {"clusters": 50})):
    vectors = None if method == "random" else pool_emb
    inputs = [cands["pool"].id] if vectors is None else [cands["pool"].id, pool_emb.id]
    for seed in (0, 1, 2):
        sel = reuse(existing("curate.select", inputs, method=method, n_max=1000, seed=seed),
                    lambda: curate.select(m, cands["pool"], vectors, method=method, n_max=1000, seed=seed, **extra))
        print("select", method, seed, sel.id, flush=True)
