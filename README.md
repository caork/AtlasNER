# AtlasNER

JPT (Joint Prompt Tuning) style discriminative NER system built on Qwen3.5-2B, with CRF decoding and definition-guided entity typing.

**Test F1 = 0.905** on CoNLL2003 (PER 0.965 / LOC 0.915 / ORG 0.885 / MISC 0.790)

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

> Requires `transformers>=4.57.0` (for Qwen3.5 support), `torch>=2.5`, `jieba`

### Start API Server

```bash
python scripts/serve.py --run-dir outputs/optimized/full_jpt --port 8000
```

Model loads in ~30s, then the server is ready at `http://localhost:8000`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ner` | POST | Extract entities from text |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

### curl Examples

**English:**

```bash
curl -s http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Barack Obama visited the United Nations headquarters in New York City."}' \
  | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

Response:

```json
{
  "tokens": ["Barack", "Obama", "visited", "the", "United", "Nations", "headquarters", "in", "New", "York", "City", "."],
  "tags": ["B-PER", "E-PER", "O", "O", "B-ORG", "E-ORG", "O", "O", "B-LOC", "I-LOC", "E-LOC", "O"],
  "entities": [
    {"type": "PER", "start": 0, "end": 2, "text": "Barack Obama"},
    {"type": "ORG", "start": 4, "end": 6, "text": "United Nations"},
    {"type": "LOC", "start": 8, "end": 11, "text": "New York City"}
  ],
  "latency_ms": 2635.7
}
```

**Chinese (uses jieba word segmentation):**

```bash
curl -s http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "腾讯创始人马化腾与阿里巴巴集团董事长蔡崇信在北京参加了中国国际进口博览会。"}' \
  | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

Response:

```json
{
  "tokens": ["腾讯", "创始人", "马化腾", "与", "阿里巴巴", "集团", "董事长", "蔡", "崇信", "在", "北京", "参加", "了", "中国", "国际", "进口", "博览会", "。"],
  "entities": [
    {"type": "ORG", "start": 0, "end": 1, "text": "腾讯"},
    {"type": "PER", "start": 2, "end": 3, "text": "马化腾"},
    {"type": "ORG", "start": 4, "end": 5, "text": "阿里巴巴"},
    {"type": "PER", "start": 7, "end": 9, "text": "蔡崇信"},
    {"type": "LOC", "start": 10, "end": 11, "text": "北京"},
    {"type": "MISC", "start": 13, "end": 17, "text": "中国国际进口博览会"}
  ],
  "latency_ms": 644.7
}
```

**Mixed Chinese-English:**

```bash
curl -s http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "小米公司CEO雷军宣布在上海建设第二座智能工厂，与比亚迪和宁德时代展开合作。"}' \
  | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"
```

Response:

```json
{
  "entities": [
    {"type": "ORG", "text": "小米"},
    {"type": "PER", "text": "雷军"},
    {"type": "LOC", "text": "上海"},
    {"type": "ORG", "text": "比亚迪"},
    {"type": "ORG", "text": "宁德时代"}
  ]
}
```

### CLI Inference (no server needed)

```bash
python scripts/predict.py \
  --run-dir outputs/optimized/full_jpt \
  --text "Apple CEO Tim Cook met EU Commissioner Thierry Breton in Brussels."
```

## Supported Entity Types

The model is trained on CoNLL2003 and recognizes 4 entity types:

| Type | Description | Examples |
|------|-------------|----------|
| **PER** | Person names | Barack Obama, 马化腾, Tim Cook |
| **ORG** | Organizations | United Nations, 腾讯, Microsoft |
| **LOC** | Locations | New York City, 北京, Brussels |
| **MISC** | Miscellaneous named entities | Digital Markets Act, 中国国际进口博览会 |

> **Note:** Chinese NER works via Qwen3.5 backbone generalization (the model was only trained on English data). For domain-specific entity types (e.g., telecom metrics, medical terms), retraining with custom data and entity definitions is needed.

## Training

### Default training (full optimized pipeline)

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --experiment configs/experiments/full_jpt.yaml \
  --override configs/runtime/optimized_mps.yaml
```

### On Apple Silicon (16GB)

The optimized config (`configs/runtime/optimized_mps.yaml`) is tuned for 16GB unified memory:

- `batch_size: 1`, `eval_batch_size: 2`
- `gradient_checkpointing: true`
- `bfloat16` precision
- `grad_accum_steps: 8` (effective batch size = 8)

### Training Results

| Config | Data | Test Entity F1 |
|--------|------|----------------|
| Baseline (500 examples, no CRF) | 500 | 0.584 |
| **Optimized (full data, CRF)** | **14,041** | **0.905** |

Per-entity test F1 (optimized):

| PER | LOC | ORG | MISC |
|-----|-----|-----|------|
| 0.965 | 0.915 | 0.885 | 0.790 |

## Architecture

### Model Components

```
Input: "Barack Obama visited Paris"
  ↓
[x SEP x]  ← JPT dual-pass: first pass for context, second pass for labeling
  ↓
Qwen3.5-2B backbone (frozen, bfloat16)
  + LoRA adapters (q/k/v/o_proj + gate/up/down_proj, rank=16)
  ↓
Multi-layer aggregation (learned weighted sum of last 4 layers)
  ↓
Token projector MLP → token representations
Entity projector MLP → label definition representations
  ↓
Bilinear classifier: token_repr × W × label_repr
  ↓
CRF layer (BIOES transition constraints + Viterbi decoding)
  ↓
Output: B-PER E-PER O S-LOC
```

### Key Optimizations

| Feature | Description |
|---------|-------------|
| **CRF layer** | BIOES-constrained transitions, Viterbi decode — enforces valid tag sequences |
| **Expanded LoRA** | 96 adapters (attention + MLP projections) — 10.9M trainable params |
| **Multi-layer aggregation** | Learned weighted sum of last 4 hidden layers |
| **Definition embeddings** | Natural language entity type descriptions encoded by backbone |
| **Focal loss / label smoothing** | Available for non-CRF training |

## Project Structure

```
configs/
  base.yaml                          # Base config
  experiments/                        # Experiment variants (full_jpt, single_pass, etc.)
  runtime/                            # Device/scale overrides (optimized_mps, stable_mps, etc.)
scripts/
  train.py                            # Training entry point
  predict.py                          # CLI inference
  serve.py                            # FastAPI server
  cache_definitions.py                # Precompute definition embeddings
src/atlas_ner/
  data/
    dataset.py                        # Dataset loading, feature building, collation
    definitions.py                    # Entity type natural language definitions
    schemes.py                        # BIO/BIOES tagging scheme conversion
  modeling/
    jpt.py                            # JPTNERModel, backbone loading, LayerWeightedSum
    crf.py                            # CRF with BIOES constraints and Viterbi decoding
    lora.py                           # LoRA adapter injection
  losses.py                           # Cross-entropy, focal loss, label smoothing
  metrics.py                          # Token-level and entity-level F1
  trainer.py                          # Training loop, evaluation, checkpointing
```

## Outputs

Each training run writes to `output_dir/`:

- `resolved_config.yaml` — merged config
- `label_vocab.json` — label names, entity types, tag definitions
- `summary_metrics.json` — final validation + test metrics + training loss curve
- `best/checkpoint.pt` — best validation F1 checkpoint
- `last/checkpoint.pt` — final checkpoint

## Memory Requirements

| Component | Size |
|-----------|------|
| Qwen3.5-2B backbone (bfloat16) | ~5.1 GB |
| LoRA adapters (float32) | ~42 MB |
| Classification head | ~18 MB |
| CRF + LayerWeightedSum | ~2 KB |
| **Inference total** | **~8-9 GB** |
| **Training total** (with gradient checkpointing) | **~10-12 GB** |
