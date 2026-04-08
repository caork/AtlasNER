"""Dataset loading and feature preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from atlas_ner.data.definitions import build_tag_definitions
from atlas_ner.data.schemes import canonical_entity_types, convert_tag_scheme


@dataclass
class LabelVocab:
    label_names: list[str]
    entity_types: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    tag_definitions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_names": self.label_names,
            "entity_types": self.entity_types,
            "tag_definitions": self.tag_definitions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelVocab":
        label_names = list(payload["label_names"])
        return cls(
            label_names=label_names,
            entity_types=list(payload["entity_types"]),
            label_to_id={name: idx for idx, name in enumerate(label_names)},
            id_to_label={idx: name for idx, name in enumerate(label_names)},
            tag_definitions=dict(payload["tag_definitions"]),
        )


def load_label_vocab(path: str | Path) -> LabelVocab:
    with Path(path).open("r", encoding="utf-8") as handle:
        return LabelVocab.from_dict(json.load(handle))


def save_label_vocab(label_vocab: LabelVocab, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(label_vocab.to_dict(), handle, ensure_ascii=False, indent=2)


class PreparedNERDataset(torch.utils.data.Dataset):
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


class NERDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)
        padded: dict[str, Any] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "word_start_mask": [],
            "prediction_positions": [],
            "word_labels": [],
            "tokens": [],
        }

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            padded["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad_len)
            padded["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            padded["labels"].append(item["labels"] + [-100] * pad_len)
            padded["word_start_mask"].append(item["word_start_mask"] + [0] * pad_len)
            padded["prediction_positions"].append(item["prediction_positions"])
            padded["word_labels"].append(item["word_labels"])
            padded["tokens"].append(item["tokens"])

        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
            "word_start_mask": torch.tensor(padded["word_start_mask"], dtype=torch.bool),
            "prediction_positions": padded["prediction_positions"],
            "word_labels": padded["word_labels"],
            "tokens": padded["tokens"],
        }


def resolve_separator_token_id(
    tokenizer: PreTrainedTokenizerBase,
    separator_token: str | None,
) -> int:
    if separator_token:
        token_id = tokenizer.convert_tokens_to_ids(separator_token)
        if token_id != tokenizer.unk_token_id:
            return int(token_id)
    for candidate in (tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if candidate is not None:
            return int(candidate)
    raise ValueError("Could not resolve a separator token id for JPT input construction.")


def word_to_subword_ids(
    tokenizer: PreTrainedTokenizerBase,
    words: list[str],
) -> list[list[int]]:
    pieces: list[list[int]] = []
    unk_id = tokenizer.unk_token_id
    for word in words:
        tokenized = tokenizer(word, add_special_tokens=False)
        input_ids = list(tokenized["input_ids"])
        if not input_ids:
            if unk_id is None:
                raise ValueError(f"Tokenizer produced no ids for word {word!r} and has no unk token.")
            input_ids = [int(unk_id)]
        pieces.append([int(token_id) for token_id in input_ids])
    return pieces


def truncate_word_pieces(
    pieces: list[list[int]],
    max_length: int,
    use_jpt: bool,
) -> int:
    total = 0
    kept = 0
    for word_pieces in pieces:
        next_total = total + len(word_pieces)
        candidate_total = next_total * (2 if use_jpt else 1) + (1 if use_jpt else 0)
        if candidate_total > max_length:
            break
        total = next_total
        kept += 1
    return max(kept, 1 if pieces else 0)


def build_feature(
    tokenizer: PreTrainedTokenizerBase,
    words: list[str],
    label_ids: list[int],
    max_length: int,
    use_jpt: bool,
    separator_token_id: int,
) -> dict[str, Any]:
    pieces = word_to_subword_ids(tokenizer, words)
    keep_words = truncate_word_pieces(pieces=pieces, max_length=max_length, use_jpt=use_jpt)
    words = words[:keep_words]
    label_ids = label_ids[:keep_words]
    pieces = pieces[:keep_words]
    if pieces:
        per_pass_budget = max_length if not use_jpt else max((max_length - 1) // 2, 1)
        current_tokens = sum(len(token_ids) for token_ids in pieces)
        if current_tokens > per_pass_budget:
            overflow = current_tokens - per_pass_budget
            pieces[-1] = pieces[-1][:-overflow] if overflow < len(pieces[-1]) else pieces[-1][:1]

    first_pass = [token_id for token_ids in pieces for token_id in token_ids]
    second_pass = [token_id for token_ids in pieces for token_id in token_ids]
    input_ids = list(first_pass)
    prediction_offset = 0

    if use_jpt:
        input_ids.append(separator_token_id)
        prediction_offset = len(input_ids)
        input_ids.extend(second_pass)
    else:
        prediction_offset = 0

    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)
    word_start_mask = [0] * len(input_ids)
    prediction_positions: list[int] = []

    cursor = prediction_offset
    for token_ids, label_id in zip(pieces, label_ids, strict=True):
        prediction_positions.append(cursor)
        labels[cursor] = label_id
        word_start_mask[cursor] = 1
        cursor += len(token_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "word_start_mask": word_start_mask,
        "prediction_positions": prediction_positions,
        "word_labels": label_ids,
        "tokens": words,
    }


def infer_original_label_names(dataset: Dataset, label_column: str) -> list[str]:
    feature = dataset.features[label_column]
    names = getattr(getattr(feature, "feature", None), "names", None)
    if names is None:
        raise ValueError(f"Could not infer label names from dataset column {label_column!r}")
    return list(names)


def build_label_vocab(
    raw_train_split: Dataset,
    dataset_name: str,
    label_column: str,
    scheme: str,
    use_label_name_only: bool,
) -> LabelVocab:
    original_label_names = infer_original_label_names(raw_train_split, label_column)
    seen_tags: list[str] = []
    for example in raw_train_split:
        tags = [original_label_names[label_id] for label_id in example[label_column]]
        seen_tags.extend(convert_tag_scheme(tags, scheme))

    entity_types = canonical_entity_types(seen_tags)
    label_names = ["O"]
    tag_prefixes = ["B", "I"] if scheme.upper() == "BIO" else ["B", "I", "E", "S"]
    for prefix in tag_prefixes:
        for entity_type in entity_types:
            label_names.append(f"{prefix}-{entity_type}")

    tag_definitions = build_tag_definitions(
        label_names=label_names,
        dataset_name=dataset_name,
        use_label_name_only=use_label_name_only,
    )
    return LabelVocab(
        label_names=label_names,
        entity_types=entity_types,
        label_to_id={name: idx for idx, name in enumerate(label_names)},
        id_to_label={idx: name for idx, name in enumerate(label_names)},
        tag_definitions=tag_definitions,
    )


def convert_raw_tags_to_ids(
    raw_label_ids: list[int],
    original_label_names: list[str],
    label_vocab: LabelVocab,
    scheme: str,
) -> list[int]:
    original_tags = [original_label_names[label_id] for label_id in raw_label_ids]
    converted_tags = convert_tag_scheme(original_tags, scheme)
    return [label_vocab.label_to_id[tag] for tag in converted_tags]


def load_public_dataset(config: dict[str, Any]) -> DatasetDict:
    dataset_kwargs = {}
    dataset_config_name = config["data"].get("dataset_config_name")
    if dataset_config_name:
        dataset_kwargs["name"] = dataset_config_name
    dataset_kwargs["trust_remote_code"] = True
    return load_dataset(config["data"]["dataset_name"], **dataset_kwargs)


def prepare_datasets(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[dict[str, PreparedNERDataset], LabelVocab]:
    dataset_dict = load_public_dataset(config)
    data_config = config["data"]
    raw_train = dataset_dict[data_config["train_split"]]
    label_vocab = build_label_vocab(
        raw_train_split=raw_train,
        dataset_name=data_config["dataset_name"],
        label_column=data_config["label_column"],
        scheme=data_config["scheme"],
        use_label_name_only=config["definitions"]["use_label_name_only"],
    )
    original_label_names = infer_original_label_names(raw_train, data_config["label_column"])
    separator_token_id = resolve_separator_token_id(
        tokenizer=tokenizer,
        separator_token=config["model"].get("separator_token"),
    )

    prepared: dict[str, PreparedNERDataset] = {}
    max_examples = data_config.get("max_examples")
    for split_name in (
        data_config["train_split"],
        data_config["validation_split"],
        data_config["test_split"],
    ):
        raw_split = dataset_dict[split_name]
        features: list[dict[str, Any]] = []
        iterable = raw_split.select(range(min(max_examples, len(raw_split)))) if max_examples else raw_split
        for example in tqdm(iterable, desc=f"Preparing {split_name}", leave=False):
            tokens = list(example[data_config["text_column"]])
            label_ids = convert_raw_tags_to_ids(
                raw_label_ids=list(example[data_config["label_column"]]),
                original_label_names=original_label_names,
                label_vocab=label_vocab,
                scheme=data_config["scheme"],
            )
            features.append(
                build_feature(
                    tokenizer=tokenizer,
                    words=tokens,
                    label_ids=label_ids,
                    max_length=int(config["model"]["max_length"]),
                    use_jpt=bool(config["model"]["use_jpt"]),
                    separator_token_id=separator_token_id,
                )
            )
        prepared[split_name] = PreparedNERDataset(features)
    return prepared, label_vocab
