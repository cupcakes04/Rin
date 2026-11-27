"""M4 step 1: the pool and val chains in workspace/store."""
import time
from datetime import datetime
from rin import Master, core, curate
from rin.servants import dinov3

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REV = "ea8dc2863c51be0a264bab82070e3e8836b02d51"

def meva_name(path):
    day, start, _end, site, camera, _rest = path.name.split(".", 5)
    return {"site": site, "camera": camera,
            "started_at": datetime.strptime(f"{day} {start}", "%Y-%m-%d %H-%M-%S")}

m = Master("workspace/store")
pool = m.run("20261126-113143-core.ingest-7de8")
val = core.ingest(m, "workspace/media/meva/val", split="val", describe=meva_name)
print("val ingest", val.id, len(val.media()), "media", flush=True)

for name, ing in (("pool", pool), ("val", val)):
    t = time.time()
    frames = core.sample(m, ing, fps=2)
    scores = curate.score_frames(m, frames, ["motion", "blur", "dark", "bright"])
    kept = curate.keep(m, scores, where="motion >= 0.002")
    print(name, "sampled", frames.id, len(frames), "kept", kept.id, len(kept), flush=True)
    emb = dinov3.embed(m, kept, model=MODEL, revision=REV)
    cands = curate.dedupe(m, kept, emb, cosine_max=0.99)
    print(name, "emb", emb.id, "cands", cands.id, len(cands), f"{time.time() - t:.0f} s", flush=True)
