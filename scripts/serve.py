#!/usr/bin/env python
"""FastAPI server for AtlasNER inference."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas_ner.config import load_yaml
from atlas_ner.data.dataset import build_feature, load_label_vocab, resolve_separator_token_id
from atlas_ner.data.schemes import tags_to_spans
from atlas_ner.modeling.jpt import load_model_from_checkpoint, load_tokenizer
from atlas_ner.trainer import get_device


# ---------------------------------------------------------------------------
# Global model holder (loaded once at startup)
# ---------------------------------------------------------------------------
class ModelHolder:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.label_vocab = None
        self.config = None
        self.device = None
        self.separator_token_id = None

    def load(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        print(f"Loading model from {run_dir} ...")
        t0 = time.time()

        self.config = load_yaml(run_dir / "resolved_config.yaml")
        self.label_vocab = load_label_vocab(run_dir / "label_vocab.json")
        self.tokenizer = load_tokenizer(self.config["model"])
        self.separator_token_id = resolve_separator_token_id(
            tokenizer=self.tokenizer,
            separator_token=self.config["model"].get("separator_token"),
        )
        model, _ = load_model_from_checkpoint(
            checkpoint_dir=run_dir / "best",
            model_config=self.config["model"],
            training_config=self.config["training"],
            num_labels=len(self.label_vocab.label_names),
            map_location="cpu",
            label_names=self.label_vocab.label_names,
        )
        self.device = get_device()
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"Model loaded in {time.time() - t0:.1f}s  (device={self.device})")


holder = ModelHolder()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class NERRequest(BaseModel):
    text: str = Field(..., description="Raw text to extract entities from", min_length=1)


class Entity(BaseModel):
    type: str
    start: int
    end: int
    text: str


class NERResponse(BaseModel):
    tokens: list[str]
    tags: list[str]
    entities: list[Entity]
    latency_ms: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AtlasNER", version="1.0.0", description="Named Entity Recognition API")


_CJK_RANGES = (
    r"\u2E80-\u9FFF"    # CJK Unified + Radicals + Kangxi + misc
    r"\uF900-\uFAFF"    # CJK Compatibility Ideographs
    r"\U00020000-\U0002FA1F"  # CJK Extension B-F + Supplements
)
_CJK_PAT = re.compile(
    rf"[{_CJK_RANGES}]"            # each CJK char is its own token
    r"|[a-zA-Z0-9]+(?:'[a-z]+)?"   # Latin words (don't, it's)
    r"|[^\w\s]",                    # punctuation
    flags=re.UNICODE,
)


def simple_word_tokenize(text: str) -> list[str]:
    return _CJK_PAT.findall(text)


@app.post("/ner", response_model=NERResponse)
def predict_ner(req: NERRequest):
    t0 = time.time()
    tokens = simple_word_tokenize(req.text)
    if not tokens:
        return NERResponse(tokens=[], tags=[], entities=[], latency_ms=0.0)

    feature = build_feature(
        tokenizer=holder.tokenizer,
        words=tokens,
        label_ids=[holder.label_vocab.label_to_id["O"]] * len(tokens),
        max_length=int(holder.config["model"]["max_length"]),
        use_jpt=bool(holder.config["model"]["use_jpt"]),
        separator_token_id=holder.separator_token_id,
    )
    input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=holder.device)
    attention_mask = torch.tensor([feature["attention_mask"]], dtype=torch.long, device=holder.device)

    with torch.no_grad():
        pred_ids = holder.model.decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prediction_positions=[feature["prediction_positions"]],
        )[0]

    pred_tags = [holder.label_vocab.id_to_label[i] for i in pred_ids]
    entities = [
        Entity(type=etype, start=s, end=e, text=" ".join(tokens[s:e]))
        for etype, s, e in tags_to_spans(pred_tags)
    ]
    return NERResponse(
        tokens=tokens,
        tags=pred_tags,
        entities=entities,
        latency_ms=round((time.time() - t0) * 1000, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": holder.model is not None}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/optimized/full_jpt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    holder.load(args.run_dir)
    uvicorn.run(app, host=args.host, port=args.port)
