# Lessons from Rin v1

> From reading `dev/rin_v1/` in full (2026-11-23). v1 ids (ADR-n, CR-n, debt An) refer to `dev/rin_v1/.claude/plans/`.

## Size, for scale

The v1 SDK had about 21k lines of Python; Studio added 7.6k more, plus a web app.

| Package | Lines | What |
|---|---|---|
| `rin/services` | 8.7k | 12 FastAPI services |
| `rin/contracts` | 4.0k | Request types, HTTP clients, per-service configs |
| `rin/data` | 5.8k | Catalog, staging, transforms, exporters |
| `rin/codecs` + `storage` + `cli` | 2.6k | Time, features, media, trackers |

About 60% of the SDK was the layer between the data and the models. After milestone M4, the first end-to-end workflow (video → teacher → export → student → evaluation) had still never run.

## Keep: these were right

| What | v1 source | Now |
|---|---|---|
| Records split by dimension: per-image class, action span, image instance, per-frame detection, track | `contracts/primitives/records.py` | The basis of the new data model |
| An action belongs to a track (actor) or to the whole scene; a whole-clip class is an action spanning the clip | ADR 1 | Keep |
| Geometry normalised to [0, 1]; pixels only at the edges | ADR 4 | D-10 |
| Time from the container's real per-frame timestamps; never `t × average fps` | ADR 5 | D-09 |
| Context padding of 25–40% around actor boxes | ADR 6 | [vlm_window_actions](../ideas/vlm_window_actions.md) |
| Exporters write native formats, no generic dataloader | ADR 2 | D-12 |
| Keypoints matched by name | ADR 16 | D-11 |
| Models label, people judge; reject incomplete spans so their frames are never exported | direction §4, CR-27 | D-02 (judgments) |
| Several prompt texts may map to one class | CR-26 | Belongs in the SAM3 servant's arguments |
| Vendors stay untouched | ADR 13 | Rule in CLAUDE.md |

## Never repeat: the evidence

| Mistake | What it caused |
|---|---|
| **One interface for all models** (3 pillars × `ground_image/video/frames`, `require_features` / `require_spatial` flags) | `Servant` grew 5 near-identical ~60-line methods. CR-31: Sapiens pose records dropped by the label translation. CR-33: setup schemas marked required fields optional. ADR 7 (base64 features in JSON) existed only to push tensors over HTTP |
| **Long-lived stateful services** (PM2, ports, VRAM governor) | CR-25: SAM3 reused the previous job's prompts. Debt A6: governor races, services stuck in `stopping`. 421 lines of governor |
| **A hand-built database with version control** (mutable dicts, id counters, fork and commit tracking inherited, edited and untouched ids, revisions, etags) | CR-12. CR-32: a branch shared nested objects with main, so editing one changed the other. Debts A1–A5 and A8: resuming twice duplicated records, commits stuck halfway, undo losing HEAD's parent |
| **Float timestamps matched within a tolerance** | CR-30: clips not starting at 0.0 lost every sampled frame |
| **Silent failure** | CR-29: a bare `except` turned every tracker trial into empty output, so all 50 trials scored 0.0, and "hota" quietly meant MOTA |
| **Infrastructure before a closed loop** | Studio M0–M4: branches, jobs, governor and API, with 662 tests, while no workflow had run end to end |
| **One JSON catalog** | 2.35 s per commit at 250k records. Each ingest rewrote the whole file |

## Salvage: code worth keeping

Servants are rewritten (D-18). These pure helpers are worth reading, and copying with the user's OK, when the milestone that needs them starts:

| v1 file | What | Needed in |
|---|---|---|
| `src/rin/codecs/timecoder.py` (`read_frame_times`, `frames_at`) | Time → the frame on screen at that time, from real per-frame timestamps (VFR-correct, tested with a VFR fixture) | M1 |
| `src/rin/codecs/spatial_ops/convert.py` | RLE ↔ mask ↔ polygon, bbox from polygon or keypoints | M3–M4 |
| `src/rin/codecs/spatial_ops/calculate.py` | Normalise, denormalise, expand (context padding), IoU | M2–M4 |
| `src/rin/data/export/yolo/writer.py`, `coco/writer.py` | Known-good layouts for YOLO and COCO exports | M4 |
| `src/rin/data/transform/pooler.py` | Bilinear RoI pooling of a DINOv3 patch grid inside a box | holy grail §4, search |
| `src/rin/contracts/clients/ground/sapiens/goliath.py` | Goliath 308 keypoint names (kept in sync with the vendor config) | actors |
| `tests/sdk/data/` | VFR and CFR video fixtures | M1 |

Everything else (contracts, clients, service apps, Servant, Master, Recorder, staging, the Studio backend) is not carried over.

## The one-sentence lesson

The schema was right. The debt came from everything between the schema and the models, and from building that before proving one loop end to end.
