# AtlasNER Agents Guide

## Project Goal

This repository hosts a JPT-style discriminative named entity recognition pipeline built on top of `Qwen/Qwen3.5-2B`.

## Working Agreements

- Keep the backbone frozen unless a task explicitly asks to change the training regime.
- Preserve the core JPT setup:
  - `x [SEP] x` dual-pass input by default
  - classify only second-pass tokens
  - first subword only participates in the loss
- Prefer explicit YAML config changes over hard-coded experiment toggles.
- Treat ablation variants as first-class configs, not ad hoc script flags.
- Keep dataset-facing code generic enough to support both Hugging Face datasets and local JSONL-style corpora.

## Implementation Conventions

- Put importable code under `src/atlas_ner/`.
- Keep entrypoints under `scripts/`.
- Use `apply_patch` for manual edits.
- Avoid introducing new heavyweight dependencies when a small local implementation is enough.
- When adding metrics, report both token-level and entity-level views.

## Validation Expectations

- Run at least static validation (`python -m compileall src scripts`) after structural changes.
- If model execution is blocked by local dependency/version limits, document the blocker clearly in the final handoff.
