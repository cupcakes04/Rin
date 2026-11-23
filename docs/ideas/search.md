# Search over footage

> Status: **raw.** Partly needed by the holy grail (targeted mining).

- **By text:** EmbeddingGemma 2 puts text, images and video in one 768-d space. Query "person carrying a box" against per-frame or per-clip embeddings. The user's `gemma_emb_engine` already handles prefixes, normalisation and limits.
- **By example:** DINOv3 CLS for whole frames; pooled patch tokens inside a box for "find more objects like this one" (v1's `Pooler` did bilinear RoI pooling over patch grids, a good reference).
- **Index:** brute-force cosine similarity is fine up to roughly a million vectors on this machine. Add a vector index only when it gets slow (to be measured).
