#!/usr/bin/env python
"""Run inference with a trained AtlasNER checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas_ner.config import load_yaml
from atlas_ner.data.dataset import LabelVocab, build_feature, load_label_vocab, resolve_separator_token_id
from atlas_ner.data.schemes import tags_to_spans
from atlas_ner.modeling.jpt import load_model_from_checkpoint, load_tokenizer
from atlas_ner.trainer import get_device


def simple_word_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training run directory with config and label vocab.")
    parser.add_argument("--checkpoint-subdir", default="best", help="Checkpoint subdirectory inside run dir.")
    parser.add_argument("--text", required=True, help="Raw text to tag.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "resolved_config.yaml")
    label_vocab = load_label_vocab(run_dir / "label_vocab.json")

    tokenizer = load_tokenizer(config["model"])
    model, _ = load_model_from_checkpoint(
        checkpoint_dir=run_dir / args.checkpoint_subdir,
        model_config=config["model"],
        training_config=config["training"],
        num_labels=len(label_vocab.label_names),
        map_location="cpu",
        label_names=label_vocab.label_names,
    )
    device = get_device()
    model.to(device)
    model.eval()

    tokens = simple_word_tokenize(args.text)
    separator_token_id = resolve_separator_token_id(
        tokenizer=tokenizer,
        separator_token=config["model"].get("separator_token"),
    )
    feature = build_feature(
        tokenizer=tokenizer,
        words=tokens,
        label_ids=[label_vocab.label_to_id["O"]] * len(tokens),
        max_length=int(config["model"]["max_length"]),
        use_jpt=bool(config["model"]["use_jpt"]),
        separator_token_id=separator_token_id,
    )
    batch = {
        "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([feature["attention_mask"]], dtype=torch.long, device=device),
    }
    with torch.no_grad():
        prediction_ids = model.decode(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            prediction_positions=[feature["prediction_positions"]],
        )[0]
    prediction_tags = [label_vocab.id_to_label[label_id] for label_id in prediction_ids]
    spans = [
        {
            "type": entity_type,
            "start": start,
            "end": end,
            "text": " ".join(tokens[start:end]),
        }
        for entity_type, start, end in tags_to_spans(prediction_tags)
    ]
    payload = {
        "tokens": tokens,
        "tags": prediction_tags,
        "entities": spans,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
