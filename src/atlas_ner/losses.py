"""Loss builders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def build_class_weights(
    labels: list[list[int]],
    num_labels: int,
    class_weighting: str,
) -> torch.Tensor | None:
    if class_weighting.lower() == "none":
        return None
    counts = torch.zeros(num_labels, dtype=torch.float)
    for sequence in labels:
        for label in sequence:
            if label >= 0:
                counts[label] += 1.0
    counts = counts.clamp_min(1.0)
    if class_weighting.lower() == "balanced":
        weights = counts.sum() / (num_labels * counts)
        return weights
    raise ValueError(f"Unsupported class weighting: {class_weighting}")


def sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    return F.cross_entropy(
        flat_logits,
        flat_labels,
        weight=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def sequence_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != ignore_index
    if not valid_mask.any():
        return flat_logits.sum() * 0.0

    flat_logits = flat_logits[valid_mask]
    flat_labels = flat_labels[valid_mask]
    log_probs = F.log_softmax(flat_logits, dim=-1)
    probs = log_probs.exp()
    target_log_probs = log_probs.gather(dim=-1, index=flat_labels.unsqueeze(-1)).squeeze(-1)
    target_probs = probs.gather(dim=-1, index=flat_labels.unsqueeze(-1)).squeeze(-1)
    modulating = (1.0 - target_probs).pow(gamma)
    losses = -modulating * target_log_probs
    if class_weights is not None:
        losses = losses * class_weights.to(losses.device)[flat_labels]
    return losses.mean()


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    training_config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_type = training_config["loss_type"].lower()
    if class_weights is not None:
        class_weights = class_weights.to(logits.device)
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    if loss_type == "weighted_ce":
        return sequence_cross_entropy(
            logits=logits, labels=labels, class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    if loss_type == "focal":
        return sequence_focal_loss(
            logits=logits,
            labels=labels,
            gamma=float(training_config["focal_gamma"]),
            class_weights=class_weights,
        )
    raise ValueError(f"Unsupported loss type: {loss_type}")
