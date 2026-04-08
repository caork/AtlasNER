# Just Pass Twice (JPT) — Markdown Technical Notes for Implementation

> This file is a **structured Markdown summary and implementation note**, not a verbatim conversion of the original paper. It is intended to be checked into a code repo alongside an implementation.

## Paper metadata

- **Title:** Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER
- **Authors:** Ahmed Ewais, Ahmed Hashish, Amr Ali
- **Affiliation:** WitnessAI
- **Project / paper URL:** https://witness.ai/witnessai-research/just-pass-twice

---

## 1. Problem statement

The paper addresses a core limitation of decoder-only LLMs for NER:

- standard NER is usually framed as **token classification**
- token classification needs **future context** to disambiguate entity labels
- decoder-only LLMs use **causal attention**, so each token can only attend to itself and earlier tokens
- this makes straightforward token classification weak when the correct label depends on right context

Example intuition from the paper:

- in `Paris released a new album`, the token `Paris` should be classified as **person**, not **location**
- under causal masking, `Paris` cannot see `released a new album` when it is encoded

The paper argues that existing LLM NER systems often switch to **generative extraction**, but that introduces:

- slow autoregressive decoding
- hallucinated entities
- format instability / parsing errors

---

## 2. Core idea: Just Pass Twice (JPT)

### 2.1 Input duplication trick

JPT duplicates the input sequence:

```text
x [SEP] x
```

The key insight is:

- tokens in the **first pass** are encoded with normal causal masking
- tokens in the **second pass** can attend backward to the entire first pass
- therefore each token in the second pass effectively has access to the whole sentence
- the model still uses the original causal attention implementation, so **no backbone architectural change is required**

Only the **second-pass token states** are used for classification.

### 2.2 Why it is fast

Although the sequence length is doubled, the paper emphasizes that this still happens in the **parallel prefill stage**, not during autoregressive token-by-token decoding. So the system avoids the main latency cost of generative extraction.

---

## 3. Model architecture

The paper uses a **frozen Qwen3 backbone** plus lightweight trainable modules.

### 3.1 Backbone

- Frozen base LLM:
  - `Qwen3-4B`
  - `Qwen3-8B`

### 3.2 Trainable components

1. **LoRA adapters** on attention projections
2. **Token projection MLP**
3. **Entity projection MLP**
4. **Bilinear classifier**

### 3.3 High-level computation flow

```text
Input: x [SEP] x
        |
Frozen Qwen3 backbone + LoRA
        |
Take hidden states from second pass only
        |
Token Projection MLP
        |
Shared token space (d_p = 256)

Entity type definitions
        |
Text encoder
        |
Entity Projection MLP
        |
Shared entity space (d_p = 256)

Token/entity matching via bilinear classifier
```

---

## 4. Definition-guided entity typing

A major component of JPT is that entity types are represented using **natural-language definitions**, not just label names.

Instead of only using labels like:

- `PERSON`
- `ORG`
- `LOC`

JPT encodes textual definitions such as:

- `A person, including individuals, artists, public figures...`
- `An organization, institution, company, team...`

The paper uses definitions in **two channels**:

1. **Classifier-side definitions**
   - entity definitions are embedded with a text encoder
   - then projected to the shared representation space
   - token states are matched against these definition embeddings

2. **Prompt-side definitions**
   - definitions are also included in the LLM input prompt to guide token representations via attention

This is meant to improve **zero-shot transfer** and make label semantics more controllable.

The paper also notes that entity definition embeddings can be **precomputed and cached** at inference time.

---

## 5. Classification head

JPT does **not** use a plain linear token classifier as the main formulation.

Instead, it projects token representations and entity-definition representations into a shared space, then scores them with a **bilinear classifier**.

A simplified form is:

```math
score(i, j) = t_i^T W e_j
```

Where:

- `t_i` = projected token representation for token `i`
- `e_j` = projected entity-definition embedding for label `j`
- `W` = learned bilinear parameter matrix

This makes the model naturally compatible with flexible or zero-shot label sets.

---

## 6. Training setup from the paper

### 6.1 Optimization

The paper reports:

- optimizer: **AdamW**
- learning rate: **5e-5**
- effective batch size: **8**
- epochs: **5**

### 6.2 Base models and dimensions

The appendix reports:

#### JPT-4B
- base model: `Qwen3-4B`
- hidden dim: `2560`
- LoRA rank / alpha: `32 / 64`
- LoRA target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- token projection MLP: `d_llm -> 1024 -> d_p`
- entity projection MLP: `d_enc -> 1024 -> d_p`
- shared dimension `d_p = 256`

#### JPT-8B
- base model: `Qwen3-8B`
- hidden dim: `4096`
- LoRA rank / alpha: `128 / 256`
- LoRA target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- token projection MLP: `d_llm -> 1024 -> 512 -> d_p`
- entity projection MLP: `d_enc -> 1024 -> 512 -> d_p`
- shared dimension `d_p = 256`

### 6.3 Entity encoder

The paper states that entity embeddings use:

- `Qwen3-Embedding-8B`

### 6.4 Training data

The paper says JPT is trained on an **in-house Wikipedia-derived NER dataset** with **no overlap** with the evaluation benchmarks.

This matters because the reported results are positioned as genuine **zero-shot transfer**, rather than benchmark-specific supervised fine-tuning.

---

## 7. Reported results

### 7.1 CrossNER + MIT zero-shot benchmark

The paper reports the following average F1 across CrossNER and MIT benchmarks:

| Model | Average F1 |
|---|---:|
| JPT-4B | 70.9 |
| JPT-8B | 74.1 |

The paper states that **JPT-8B improves over the strongest baseline (SaM) by +7.9 F1 on average**.

The table snippet visible from the paper shows:

| Model | AI | Literature | Music | Politics | Science | Movie | Restaurant | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JPT-4B | 68.3 | 73.7 | 84.1 | 76.4 | 69.5 | 60.7 | 63.4 | 70.9 |
| JPT-8B | 71.9 | 72.2 | 85.3 | 77.0 | 71.3 | 76.5 | 64.4 | 74.1 |

### 7.2 Extended benchmark over 20 datasets

The paper reports extended zero-shot F1 over 20 NER datasets.

Visible values from the paper include:

| Dataset | UniNER-7B | GLiNER-L | JPT-4B |
|---|---:|---:|---:|
| ACE05 | 36.9 | 27.3 | 44.6 |
| AnatEM | 25.1 | 33.3 | 37.2 |
| bc2gm | 46.2 | 47.9 | 54.8 |
| bc4chemd | 47.9 | 43.1 | 53.3 |
| bc5cdr | 68.0 | 66.4 | 70.4 |
| Broad Twitter | 67.9 | 61.2 | 71.2 |
| CoNLL03 | 72.2 | 64.6 | 78.1 |
| FabNER | 24.8 | 23.6 | 27.8 |
| FindVehicle | 22.2 | 41.9 | 42.3 |
| GENIA | 54.1 | 55.5 | 50.8 |
| HarveyNER | 18.2 | 22.7 | 27.7 |
| MIT Movie | 42.4 | 57.2 | 73.4 |
| MIT Restaurant | 31.7 | 42.9 | 61.9 |
| MultiNERD | 59.3 | 59.7 | 65.7 |
| NCBI | 60.4 | 61.9 | 68.7 |
| OntoNotes | 27.8 | 32.2 | 43.1 |
| PolyglotNER | 41.8 | 42.9 | 47.4 |
| TweetNER7 | 42.7 | 41.4 | 49.7 |
| WikiANN | 55.4 | 58.9 | 64.7 |
| WikiNeural | 69.2 | 71.8 | 77.3 |
| **Average** | **45.7** | **47.8** | **55.5** |

The paper notes that JPT-4B outperforms the listed baselines on **19 of 20** datasets.

### 7.3 Speed claim

The paper claims JPT is **over 20× faster than comparable generative methods**.

---

## 8. Paper takeaways for our implementation

If we are implementing a JPT-style system with `Qwen/Qwen3.5-2B`, the most faithful parts to reproduce are:

1. **Double-pass input**: `x [SEP] x`
2. **Classify only the second pass**
3. **Frozen backbone**
4. **LoRA only on attention projections**
5. **Token projection MLP**
6. **Entity projection MLP**
7. **Definition-guided typing**
8. **Bilinear token-label scoring**

### What will differ from the paper in our repo

Our implementation will differ from the original paper in at least these ways:

- base model: `Qwen/Qwen3.5-2B` instead of `Qwen3-4B` / `Qwen3-8B`
- likely smaller LoRA rank than the paper’s 8B setting
- likely smaller max sequence length in first-pass experiments
- likely public benchmark data instead of the paper’s internal Wikipedia-derived training corpus

So our repo should describe itself as:

> a **JPT-style reproduction / adaptation**, not an exact checkpoint reproduction of the published paper.

---

## 9. Recommended repo README wording

You can paste something like this into your implementation repo:

```md
This repository implements a JPT-style discriminative NER pipeline inspired by the paper *Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER*.

The original paper uses frozen Qwen3-4B / Qwen3-8B backbones, LoRA adapters on attention projections, token/entity projection MLPs, and a bilinear classifier with definition-guided entity typing.

This repo adapts the method to `Qwen/Qwen3.5-2B` as the base model for engineering reproduction and experimentation.

It is a method-level reproduction, not an exact replication of the original paper’s released checkpoints, training corpus, or reported benchmark numbers.
```

---

## 10. Recommended experiment matrix for reproduction

### A. Full JPT
- double pass
- definitions enabled
- bilinear classifier
- LoRA enabled

### B. No definitions
- double pass
- no definition embeddings
- fixed label embeddings or direct classifier

### C. Single pass
- `x` only
- otherwise same setup
- tests whether the JPT duplication trick is doing real work

### D. Linear instead of bilinear
- replace token-definition bilinear scoring with a standard linear classifier

This matches the implementation ablations we want in the training pipeline.

---

## 11. Notes for the Codex implementation

### Token labeling
Use BIO or BIOES labels, but only compute loss on the **first subword of each original word**.

### Definition caching
Entity definition embeddings should be cached so inference does not repeatedly re-encode them.

### Losses
The implementation should support:

- weighted cross entropy
- focal loss

The paper mentions reweighting to address the dominance of the `O` class.

### Metrics
The implementation should output:

- entity-level micro F1
- token F1
- per-label F1

---

## 12. Minimal pseudocode

```python
# Build JPT input
full_input = tokens + [sep_token] + tokens

# Forward through frozen backbone + LoRA
hidden = model(full_input)

# Slice second-pass positions only
hidden_second = hidden[:, second_pass_positions, :]

# Project token representations
z_tok = token_mlp(hidden_second)

# Load cached entity definition embeddings
z_ent = entity_mlp(entity_definition_embeddings)

# Bilinear scores
scores = bilinear(z_tok, z_ent)

# Compute loss only on valid labeled first-subword positions
loss = criterion(scores[loss_mask], labels[loss_mask])
```

---

## 13. Limitations to keep in mind

- The paper’s strongest public numbers are on `Qwen3-4B` and `Qwen3-8B`, not on `Qwen3.5-2B`.
- The paper uses an internal Wikipedia-derived training set that is not the same as typical public NER training corpora.
- The full zero-shot setup depends heavily on the quality of entity definitions.
- The method is especially attractive when you want to preserve LLM knowledge while avoiding generative extraction latency.

---

## 14. References

1. WitnessAI paper / PDF: https://witness.ai/witnessai-research/just-pass-twice
2. Qwen3.5-2B model card: https://huggingface.co/Qwen/Qwen3.5-2B
3. Qwen3 technical report: https://arxiv.org/abs/2505.09388

