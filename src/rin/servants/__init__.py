"""Servants: model steps. Each folder holds one model's two halves (D-07, D-22):

- the caller (`__init__.py`), in the core environment: the real signature, argument checks,
  the frame cache, and filing the run. It imports no model framework.
- the job (`job.py`), in the model's own environment (`pyproject.toml` + `uv.lock` beside it):
  reads a spec and cached frames, writes raw outputs into the run's temporary folder, exits.
  It imports nothing from Rin; never import it from the core environment.

Shared by every servant, beside the model folders:
- `frame_cache.py`: decodes the frames a job needs into JPEGs once (D-29), so no job ever
  decodes video. Plumbing only: it says nothing about any model's inputs or outputs (D-01).
"""
