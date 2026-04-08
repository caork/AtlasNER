"""Training and evaluation utilities."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from atlas_ner.data.dataset import NERDataCollator
from atlas_ner.losses import build_class_weights
from atlas_ner.metrics import compute_all_metrics
from atlas_ner.modeling.jpt import JPTNERModel, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    pad_token_id: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=NERDataCollator(pad_token_id=pad_token_id),
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def extract_word_level_predictions(
    logits: torch.Tensor,
    prediction_positions: list[list[int]],
) -> list[list[int]]:
    predictions = logits.argmax(dim=-1)
    outputs: list[list[int]] = []
    for batch_idx, positions in enumerate(prediction_positions):
        outputs.append([int(predictions[batch_idx, pos].item()) for pos in positions])
    return outputs


@torch.no_grad()
def evaluate(
    model: JPTNERModel,
    dataloader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    all_predictions: list[list[int]] = []
    all_references: list[list[int]] = []
    losses: list[float] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            prediction_positions=batch["prediction_positions"],
            word_labels=batch["word_labels"],
        )
        predictions = model.decode_from_logits(outputs.logits, batch["prediction_positions"])
        loss_val = float(outputs.loss.item()) if outputs.loss is not None else 0.0
        if not math.isnan(loss_val):
            losses.append(loss_val)
        all_predictions.extend(predictions)
        all_references.extend(batch["word_labels"])

    metrics = compute_all_metrics(all_predictions, all_references, id_to_label=id_to_label)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_or_load_definition_features(
    model: JPTNERModel,
    tokenizer,
    label_definitions: dict[str, str],
    cache_path: str | Path | None,
    pooling: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    cache_tensor = None
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        cache_tensor = torch.load(cache_file, map_location="cpu")
    if cache_tensor is None:
        texts = [label_definitions[label_name] for label_name in label_definitions]
        cache_tensor = model.encode_texts(
            tokenizer=tokenizer,
            texts=texts,
            batch_size=batch_size,
            pooling=pooling,
            device=device,
        )
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache_tensor, cache_file)
    return cache_tensor


def train(
    model: JPTNERModel,
    tokenizer,
    datasets: dict[str, torch.utils.data.Dataset],
    id_to_label: dict[int, str],
    label_definitions: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    training_config = config["training"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model.to(device)
    definition_features = build_or_load_definition_features(
        model=model,
        tokenizer=tokenizer,
        label_definitions=label_definitions,
        cache_path=config["definitions"].get("cache_path"),
        pooling=config["model"].get("entity_pooling", "mean"),
        batch_size=int(training_config["eval_batch_size"]),
        device=device,
    )
    model.set_definition_features(definition_features)

    class_weights = build_class_weights(
        labels=[item["word_labels"] for item in datasets[config["data"]["train_split"]]],
        num_labels=len(id_to_label),
        class_weighting=training_config["class_weighting"],
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is required for batching.")

    train_loader = create_dataloader(
        dataset=datasets[config["data"]["train_split"]],
        batch_size=int(training_config["batch_size"]),
        pad_token_id=pad_token_id,
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
    )
    val_loader = create_dataloader(
        dataset=datasets[config["data"]["validation_split"]],
        batch_size=int(training_config["eval_batch_size"]),
        pad_token_id=pad_token_id,
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )
    test_loader = create_dataloader(
        dataset=datasets[config["data"]["test_split"]],
        batch_size=int(training_config["eval_batch_size"]),
        pad_token_id=pad_token_id,
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n]
    head_lr = float(training_config.get("head_learning_rate", training_config["learning_rate"]))
    print(f"  Trainable: backbone(LoRA)={sum(p.numel() for p in backbone_params):,}  head={sum(p.numel() for p in head_params):,}")
    print(f"  LR: backbone={float(training_config['learning_rate']):.1e}  head={head_lr:.1e}")
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": float(training_config["learning_rate"])},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    total_update_steps = math.ceil(len(train_loader) * int(training_config["epochs"]) / int(training_config["grad_accum_steps"]))
    warmup_steps = int(total_update_steps * float(training_config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    amp_enabled = bool(training_config["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and config["model"].get("torch_dtype") == "float16")
    amp_dtype = torch.bfloat16 if config["model"].get("torch_dtype") == "bfloat16" else torch.float16

    best_metric = float("-inf")
    global_step = 0
    train_logs: list[dict[str, Any]] = []

    for epoch in range(int(training_config["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        running_loss = 0.0

        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    class_weights=class_weights,
                    prediction_positions=batch["prediction_positions"],
                    word_labels=batch["word_labels"],
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("Training forward pass did not produce a loss.")
                loss = loss / int(training_config["grad_accum_steps"])

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item())
            if step % int(training_config["grad_accum_steps"]) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(training_config["max_grad_norm"]),
                )
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(training_config["log_every_steps"]) == 0:
                    average_loss = running_loss / max(int(training_config["log_every_steps"]), 1)
                    progress.set_postfix(loss=f"{average_loss:.4f}", gnorm=f"{float(grad_norm):.2f}")
                    train_logs.append({"step": global_step, "loss": average_loss, "grad_norm": float(grad_norm)})
                    running_loss = 0.0

                if global_step % int(training_config["eval_every_steps"]) == 0:
                    val_metrics = evaluate(model, val_loader, device=device, id_to_label=id_to_label)
                    save_json(val_metrics, output_dir / "val_metrics_latest.json")
                    if val_metrics["entity_micro_f1"] > best_metric:
                        best_metric = val_metrics["entity_micro_f1"]
                        save_checkpoint(
                            model=model,
                            output_dir=output_dir / "best",
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=global_step,
                            epoch=epoch,
                            metrics=val_metrics,
                        )

        remainder = len(train_loader) % int(training_config["grad_accum_steps"])
        if remainder != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training_config["max_grad_norm"]),
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_metrics = evaluate(model, val_loader, device=device, id_to_label=id_to_label)
        save_json(val_metrics, output_dir / f"val_metrics_epoch_{epoch + 1}.json")
        if val_metrics["entity_micro_f1"] > best_metric:
            best_metric = val_metrics["entity_micro_f1"]
            save_checkpoint(
                model=model,
                output_dir=output_dir / "best",
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                epoch=epoch,
                metrics=val_metrics,
            )

    save_checkpoint(
        model=model,
        output_dir=output_dir / "last",
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        epoch=int(training_config["epochs"]),
        metrics={},
    )
    best_checkpoint = torch.load(output_dir / "best" / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state"])
    model.to(device)

    val_metrics = evaluate(model, val_loader, device=device, id_to_label=id_to_label)
    test_metrics = evaluate(model, test_loader, device=device, id_to_label=id_to_label)
    summary = {
        "best_validation": val_metrics,
        "test": test_metrics,
        "train_logs": train_logs,
    }
    save_json(summary, output_dir / "summary_metrics.json")
    return summary
