# AtlasNER - Claude Code Guidelines

## Memory Constraints (CRITICAL)

This machine has **16GB unified memory (Apple Silicon MPS)**. Excessive memory usage causes disk swap and damages SSD lifespan.

When running training or inference on this project:
- **Always enable gradient checkpointing** (`gradient_checkpointing: true` in model config)
- **batch_size must not exceed 1** for training; **eval_batch_size must not exceed 2**
- Use `grad_accum_steps` to achieve larger effective batch sizes instead of increasing batch_size
- Use `bfloat16` dtype (not float32) to halve model weight memory
- **max_length should stay at 256 or below** unless explicitly overridden by the user
- Monitor for step-time spikes during training — spikes from ~2s to >30s indicate memory pressure/swap

## Project Structure

- `scripts/train.py` — training entry point
- `scripts/predict.py` — inference entry point
- `src/atlas_ner/` — core library (model, trainer, data, losses, metrics)
- `configs/base.yaml` — base config; `configs/experiments/` — experiment variants; `configs/runtime/` — device/scale overrides
- `Qwen3.5-2B/` — local backbone model (Qwen 3.5 2B, multimodal architecture used text-only)

## Training

Run via: `/opt/homebrew/Caskroom/miniconda/base/bin/python scripts/train.py --config configs/base.yaml --experiment <experiment.yaml> --override <runtime.yaml>`

The Python interpreter is the conda base environment at `/opt/homebrew/Caskroom/miniconda/base/bin/python`.
