#!/usr/bin/env python
"""Precompute definition embeddings for a given experiment config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from atlas_ner.config import load_config
from atlas_ner.data.dataset import prepare_datasets
from atlas_ner.modeling.jpt import JPTNERModel, load_tokenizer
from atlas_ner.trainer import build_or_load_definition_features, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment", default="configs/experiments/full_jpt.yaml")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.experiment, extra_paths=args.override)
    set_seed(int(config["seed"]))
    tokenizer = load_tokenizer(config["model"])
    _, label_vocab = prepare_datasets(config, tokenizer)
    model = JPTNERModel.from_config(
        model_config=config["model"],
        training_config=config["training"],
        num_labels=len(label_vocab.label_names),
    )
    device = get_device()
    model.to(device)
    features = build_or_load_definition_features(
        model=model,
        tokenizer=tokenizer,
        label_definitions=label_vocab.tag_definitions,
        cache_path=config["definitions"].get("cache_path"),
        pooling=config["model"].get("entity_pooling", "mean"),
        batch_size=int(config["training"]["eval_batch_size"]),
        device=device,
    )
    print(f"Cached definition features: shape={tuple(features.shape)} path={config['definitions'].get('cache_path')}")
    print(torch.norm(features, dim=-1))


if __name__ == "__main__":
    main()
