# AtlasNER

JPT-style discriminative named entity recognition training pipeline built around `Qwen/Qwen3.5-2B`.

## What is implemented

- `x [SEP] x` dual-pass input, with classification aligned to the second pass
- frozen backbone with trainable:
  - LoRA adapters on `q_proj/k_proj/v_proj/o_proj`
  - token projection MLP
  - entity projection MLP
  - bilinear classifier
- BIO and BIOES tagging
- first-subword-only supervision
- definition-guided typing with cached definition encodings
- weighted cross entropy and focal loss
- four ablations:
  - full JPT
  - no definitions
  - single pass
  - linear head instead of bilinear
- entity-level micro F1, token micro F1, and per-label entity F1

## Chosen public dataset

The default training dataset is [`conll2003`](https://huggingface.co/datasets/conll2003) via the Hugging Face `datasets` library.

Its label set is mapped into the configured tagging scheme:

- `BIO`
- `BIOES`

Natural-language definitions are provided for `PER`, `ORG`, `LOC`, and `MISC`.

## Repository layout

```text
configs/
  base.yaml
  experiments/
scripts/
  train.py
  cache_definitions.py
  predict.py
src/atlas_ner/
  data/
  modeling/
  losses.py
  metrics.py
  trainer.py
```

## Environment

This repository targets recent Transformers releases because the local backbone is `Qwen3.5`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Important version note:

- `transformers>=4.57.0` is required for `qwen3_5`

## Training

Run the default full JPT experiment:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --experiment configs/experiments/full_jpt.yaml
```

Precompute definition embeddings first if you want to avoid computing them during training:

```bash
python scripts/cache_definitions.py \
  --config configs/base.yaml \
  --experiment configs/experiments/full_jpt.yaml
```

Available experiment configs:

- `configs/experiments/full_jpt.yaml`
- `configs/experiments/no_definitions.yaml`
- `configs/experiments/single_pass.yaml`
- `configs/experiments/linear_head.yaml`

## Inference

After training, run:

```bash
python scripts/predict.py \
  --run-dir outputs/full_jpt \
  --checkpoint-subdir best \
  --text "Barack Obama visited Paris with Microsoft executives."
```

The script prints:

- tokenized words
- predicted tag sequence
- extracted entity spans

## Config notes

`configs/base.yaml` controls:

- backbone path
- LoRA rank and alpha
- projection MLP dimensions
- tagging scheme
- max input length
- optimizer and scheduler hyperparameters
- loss type and class weighting
- definition cache path

## Implementation details

### JPT input packing

By default the sequence is packed as:

```text
x [SEP] x
```

Only the second pass contributes predictions. Loss is only computed on the first subword of each original word in the prediction pass.

### Definition-guided typing

This implementation encodes label descriptions once, caches the frozen backbone features, and learns an entity projection MLP on top of those cached representations.

For BIO/BIOES tagging, the cached texts are tag-level descriptions derived from entity definitions, for example:

- `B-PER`: first token of a person entity
- `E-ORG`: final token of an organization entity

The `no_definitions` ablation switches these texts to short label-name descriptions instead of full natural-language definitions.

### Bilinear vs linear head

- `bilinear`: token projection matched against projected definition features
- `linear`: projected token states passed to a standard linear classifier

## Outputs

Each run writes under `output_dir`:

- `resolved_config.yaml`
- `label_vocab.json`
- `summary_metrics.json`
- `best/checkpoint.pt`
- `last/checkpoint.pt`

The summary includes:

- `entity_micro_f1`
- `token_micro_f1`
- `per_label_f1`

## Current caveats

- The local environment I used while scaffolding had `transformers==4.43.4`, which is too old to instantiate `qwen3_5`. The code is written against newer Transformers and should be run after upgrading dependencies.
- This implementation uses classifier-side definition guidance. It does not prepend definitions into the token input prompt.
- The default tokenizer for free-form inference is a simple regex word splitter; for benchmark evaluation, training/evaluation use the dataset's tokenized words directly.
