#!/usr/bin/env python
"""Train a JPT-style discriminative NER model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas_ner.config import load_config, save_config
from atlas_ner.data.dataset import prepare_datasets, save_label_vocab
from atlas_ner.modeling.jpt import JPTNERModel, load_tokenizer
from atlas_ner.trainer import set_seed, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--experiment", default="configs/experiments/full_jpt.yaml", help="Experiment override yaml.")
    parser.add_argument("--override", action="append", default=[], help="Additional override yaml(s), applied in order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.experiment, extra_paths=args.override)
    set_seed(int(config["seed"]))

    tokenizer = load_tokenizer(config["model"])
    datasets, label_vocab = prepare_datasets(config, tokenizer)
    model = JPTNERModel.from_config(
        model_config=config["model"],
        training_config=config["training"],
        num_labels=len(label_vocab.label_names),
        label_names=label_vocab.label_names,
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")
    save_label_vocab(label_vocab, output_dir / "label_vocab.json")

    summary = train(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        id_to_label=label_vocab.id_to_label,
        label_definitions=label_vocab.tag_definitions,
        config=config,
    )
    print(f"Training complete. Metrics saved to {output_dir / 'summary_metrics.json'}")
    print(summary["test"])


if __name__ == "__main__":
    main()
