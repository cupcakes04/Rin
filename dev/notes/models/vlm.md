# Local LLM / VLM notes (16 GB VRAM)

Notes on running instruction-tuned VLMs on one 16 GB card (RTX 4070 Ti SUPER).
The Hugging Face model cards already document usage; this file only records what they don't: **what actually fits and works on this machine**.

Status key: ✅ tested here · ⚠️ estimate (not yet tested) · ❌ tested, doesn't work

## Fit table

| Model | Weights | Fits? | Video | Notes | Status |
|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | FP8 | yes | ok | | ✅ |
| Qwen3-VL-8B-Instruct | FP8 | yes | ~10 s max | KV cache blows up beyond ~10 s at default settings | ✅ |
| Gemma 4 E2B-it | bf16 (~10 GB) | yes | frames | text, image, audio input | ⚠️ |
| Gemma 4 E4B-it | 8-bit (~8 GB) | yes | frames | bf16 is ~16 GB incl. per-layer embeddings, too tight | ⚠️ |
| Gemma 4 12B-it | 4-bit (~7 GB) | yes | yes | 8-bit is ~12 GB + KV cache, tight | ⚠️ |
| Gemma 4 26B-A4B-it (MoE) | 4-bit (~13–14 GB) | barely | no headroom | only 3.8B active, but all 25B weights must be resident | ⚠️ |
| Gemma 4 31B-it | — | no | — | | ⚠️ |

Estimates are weights only (params × bytes per param). KV cache and vision/audio encoders come on top.

## The main lever for video: visual tokens, not model size

Context use for video ≈ frames × tokens per frame.

- **Gemma 4:** visual token budget per image/frame is one of 70, 140, 280, 560, 1120. Video is capped at 60 s at 1 fps; audio at 30 s.
  - 60 frames × 280 ≈ 17k tokens; 60 frames × 70 ≈ 4k tokens.
- **Qwen3-VL:** the equivalents are `fps` and the min/max pixel limits in the processor.

Lowering these is the first thing to try before dropping to a smaller model.

## Gemma 4 facts (from the HF cards, 2026-11)

| Size | Params | Input | Context |
|---|---|---|---|
| E2B | 2.3B effective (5.1B incl. embeddings) | text, image, audio | 128K |
| E4B | 4.5B effective (8B incl. embeddings) | text, image, audio | 128K |
| 12B | 11.95B dense | text, image, audio, video | 256K |
| 26B-A4B | 25.2B total, 3.8B active (MoE) | text, image | 256K |
| 31B | 30.7B dense | text, image | 256K |

- **Usage:** `AutoProcessor` + `AutoModelForMultimodalLM`, chat template with typed content parts, `processor.parse_response()` splits thinking from the answer.
- **Thinking:** `enable_thinking=True` in `apply_chat_template`. With it off, E2B/E4B emit no thought block; the larger sizes still emit an empty one. Drop previous thinking from history, except in tool-call turns.
- **Sampling:** temperature 1.0, top_p 0.95, top_k 64.
- **Tool calling:** native. Structured (JSON-schema) output comes from the runtime (e.g. vLLM guided decoding), not from the model card.
- **Serving:** `vllm serve google/gemma-4-...` gives an OpenAI-compatible API. Official QAT Q4_0 checkpoints exist (separate HF collection); prefer them over community quants.

## WSL2 gotchas for vLLM (from v1, tested there)

Full notes: `dev/rin_v1/docs/notes/services/qwen/qwen_install.md`.

| Symptom | Fix |
|---|---|
| `pin_memory is not supported on WSL2` | `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` |
| `CUDA error: operation not supported when stream is capturing` | `enforce_eager=True` |
| FlashInfer sampler fails to compile | `export VLLM_USE_FLASHINFER_SAMPLER=0` |
| `KV cache is needed, which is larger than available` | Cap `max_model_len` and frames per request |
| Silent deadlock at start-up | `export VLLM_WORKER_MULTIPROC_METHOD=spawn` (set before launch, with the two flags above) |
| `start a new process before current process finished bootstrapping` | Caused by `spawn` above: guard the entry point with `if __name__ == "__main__":` |
| Tokenizer `AttributeError` (`all_special_tokens_extended`) | `transformers` version drift: keep vLLM in its own environment |

## How Rin uses VLMs

Actions as window classification on actor crops: [../ideas/vlm_window_actions.md](../ideas/vlm_window_actions.md). Structured output (JSON schema) comes from vLLM guided decoding.
