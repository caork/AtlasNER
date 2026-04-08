"""Evaluation metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from sklearn.metrics import f1_score, precision_recall_fscore_support

from atlas_ner.data.schemes import parse_tag, tags_to_spans


def strip_entity_type(tag: str) -> str | None:
    if tag == "O":
        return None
    _, entity_type = parse_tag(tag)
    return entity_type


def compute_token_metrics(
    predictions: list[list[int]],
    references: list[list[int]],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    flat_predictions: list[int] = []
    flat_references: list[int] = []
    valid_label_ids = [label_id for label_id, label_name in id_to_label.items() if label_name != "O"]

    for pred_seq, ref_seq in zip(predictions, references, strict=True):
        flat_predictions.extend(pred_seq)
        flat_references.extend(ref_seq)

    if not flat_predictions:
        return {"token_micro_f1": 0.0}

    token_micro_f1 = f1_score(
        flat_references,
        flat_predictions,
        labels=valid_label_ids,
        average="micro",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        flat_references,
        flat_predictions,
        labels=valid_label_ids,
        average=None,
        zero_division=0,
    )

    per_tag_f1 = {
        id_to_label[label_id]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, label_id in enumerate(valid_label_ids)
    }
    return {
        "token_micro_f1": float(token_micro_f1),
        "per_tag_token_f1": per_tag_f1,
    }


def compute_entity_metrics(
    prediction_tags: list[list[str]],
    reference_tags: list[list[str]],
) -> dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    per_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for pred_seq, ref_seq in zip(prediction_tags, reference_tags, strict=True):
        pred_spans = set(tags_to_spans(pred_seq))
        ref_spans = set(tags_to_spans(ref_seq))
        overlap = pred_spans & ref_spans
        true_positive += len(overlap)
        false_positive += len(pred_spans - ref_spans)
        false_negative += len(ref_spans - pred_spans)

        for span in overlap:
            per_label_counts[span[0]]["tp"] += 1
        for span in pred_spans - ref_spans:
            per_label_counts[span[0]]["fp"] += 1
        for span in ref_spans - pred_spans:
            per_label_counts[span[0]]["fn"] += 1

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    entity_micro_f1 = 0.0
    if precision + recall > 0:
        entity_micro_f1 = 2 * precision * recall / (precision + recall)

    per_label_f1: dict[str, dict[str, float]] = {}
    for label_name, counts in sorted(per_label_counts.items()):
        label_precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        label_recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        label_f1 = 0.0
        if label_precision + label_recall > 0:
            label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall)
        per_label_f1[label_name] = {
            "precision": float(label_precision),
            "recall": float(label_recall),
            "f1": float(label_f1),
            "support": int(counts["tp"] + counts["fn"]),
        }
    return {
        "entity_micro_f1": float(entity_micro_f1),
        "per_label_f1": per_label_f1,
    }


def compute_all_metrics(
    predictions: list[list[int]],
    references: list[list[int]],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    prediction_tags = [[id_to_label[label_id] for label_id in sequence] for sequence in predictions]
    reference_tags = [[id_to_label[label_id] for label_id in sequence] for sequence in references]
    metrics = {}
    metrics.update(compute_token_metrics(predictions, references, id_to_label))
    metrics.update(compute_entity_metrics(prediction_tags, reference_tags))
    return metrics
