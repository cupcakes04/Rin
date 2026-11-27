"""M4 step 2: SAM3 on every pool and val candidate (D-37)."""
import time
from rin import Master, curate
from rin.servants import sam3

m = Master("workspace/store")
cands = curate.union(m, [m.run("20261127-105927-curate.dedupe-6b65"), m.run("20261127-110156-curate.dedupe-f05d")])
print("union", cands.id, len(cands), flush=True)
t = time.time()
labels = sam3.concept(m, cands, prompts={"person": "person"}, checkpoint="workspace/checkpoints/sam3/sam3.pt",
                      score_floor=0.1, masks=True)
print("sam3", labels.id, f"{time.time() - t:.0f} s", flush=True)
