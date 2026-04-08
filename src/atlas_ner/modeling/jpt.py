"""JPT-style discriminative NER model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from atlas_ner.losses import compute_loss
from atlas_ner.modeling.crf import CRF, build_bioes_constraints
from atlas_ner.modeling.lora import inject_lora_adapters

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_tokenizer(model_config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["backbone_name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.unk_token_id
        if fallback_id is None:
            raise ValueError("Tokenizer has no pad/eos/unk token available.")
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(fallback_id)
    tokenizer.padding_side = "right"
    return tokenizer


def load_backbone(model_config: dict[str, Any]) -> nn.Module:
    pretrained_kwargs = {
        "pretrained_model_name_or_path": model_config["backbone_name_or_path"],
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "torch_dtype": resolve_torch_dtype(model_config.get("torch_dtype")),
    }
    config = AutoConfig.from_pretrained(
        model_config["backbone_name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    architectures = config.architectures or []
    candidates = []
    if AutoModelForImageTextToText is not None and any("ConditionalGeneration" in name for name in architectures):
        candidates.append(AutoModelForImageTextToText)
    candidates.extend([AutoModelForCausalLM, AutoModel])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return candidate.from_pretrained(**pretrained_kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Unable to load backbone model. Last error: {last_error}") from last_error


def infer_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    if config is None:
        raise ValueError("Backbone has no config to infer hidden size.")
    for attr_name in ("hidden_size",):
        value = getattr(config, attr_name, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        value = getattr(text_config, "hidden_size", None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from backbone config.")


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LayerWeightedSum(nn.Module):
    """Learned weighted sum of multiple hidden-state layers."""

    def __init__(self, num_layers: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = gamma

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        normed_weights = torch.softmax(self.weights, dim=0) * self.gamma
        stacked = torch.stack(hidden_states, dim=0)  # [L, B, S, H]
        return (normed_weights[:, None, None, None] * stacked).sum(dim=0)  # [B, S, H]


class BilinearClassifier(nn.Module):
    def __init__(self, projection_dim: int, num_labels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(projection_dim, projection_dim))
        self.bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, token_repr: torch.Tensor, label_repr: torch.Tensor) -> torch.Tensor:
        projected = torch.matmul(token_repr, self.weight)
        return torch.einsum("bld,kd->blk", projected, label_repr) + self.bias


@dataclass
class JPTModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class JPTNERModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        label_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.training_config = training_config
        projection_dim = int(model_config["projection_dim"])
        dropout = float(model_config["lora_dropout"])
        token_hidden_dims = [int(x) for x in model_config.get("token_mlp_hidden_dims", [])]
        entity_hidden_dims = [int(x) for x in model_config.get("entity_mlp_hidden_dims", [])]

        # Multi-layer aggregation
        self.num_aggregate_layers = int(model_config.get("num_aggregate_layers", 1))
        if self.num_aggregate_layers > 1:
            self.layer_weighted_sum = LayerWeightedSum(self.num_aggregate_layers)
        else:
            self.layer_weighted_sum = None

        self.token_projector = MLPProjector(hidden_size, token_hidden_dims, projection_dim, dropout)
        self.entity_projector = MLPProjector(hidden_size, entity_hidden_dims, projection_dim, dropout)
        self.classifier_type = model_config["classifier_type"].lower()
        if self.classifier_type == "bilinear":
            self.classifier = BilinearClassifier(projection_dim=projection_dim, num_labels=num_labels)
        elif self.classifier_type == "linear":
            self.classifier = nn.Linear(projection_dim, num_labels)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

        # CRF layer (optional)
        self.crf: CRF | None = None
        if model_config.get("use_crf", False) and label_names is not None:
            transition_mask, start_mask, end_mask = build_bioes_constraints(label_names)
            self.crf = CRF(
                num_labels=num_labels,
                transition_mask=transition_mask,
                start_mask=start_mask,
                end_mask=end_mask,
            )

        self.register_buffer("definition_features", torch.zeros(num_labels, hidden_size), persistent=True)

    def _projector_dtype(self) -> torch.dtype:
        return next(self.token_projector.parameters()).dtype

    @classmethod
    def from_config(
        cls,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        num_labels: int,
        label_names: list[str] | None = None,
    ) -> "JPTNERModel":
        backbone = load_backbone(model_config)
        hidden_size = infer_hidden_size(backbone)
        inject_lora_adapters(
            module=backbone,
            target_modules=list(model_config["lora_target_modules"]),
            rank=int(model_config["lora_rank"]),
            alpha=float(model_config["lora_alpha"]),
            dropout=float(model_config["lora_dropout"]),
        )
        if model_config.get("gradient_checkpointing", False):
            if hasattr(backbone, "gradient_checkpointing_enable"):
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                print("  Gradient checkpointing enabled")
        return cls(
            backbone=backbone,
            hidden_size=hidden_size,
            num_labels=num_labels,
            model_config=model_config,
            training_config=training_config,
            label_names=label_names,
        )

    def set_definition_features(self, definition_features: torch.Tensor) -> None:
        if definition_features.shape != self.definition_features.shape:
            raise ValueError(
                f"Definition features shape mismatch: expected {tuple(self.definition_features.shape)}, "
                f"got {tuple(definition_features.shape)}"
            )
        self.definition_features.copy_(definition_features)

    # ------------------------------------------------------------------
    # Helpers for CRF: extract word-level emissions and pad word labels
    # ------------------------------------------------------------------

    def _extract_word_emissions(
        self,
        logits: torch.Tensor,
        prediction_positions: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract word-level emissions from full-sequence logits.

        Returns:
            emissions: [B, max_words, num_labels]
            mask:      [B, max_words] (bool)
        """
        batch_size = logits.shape[0]
        max_words = max(len(pos) for pos in prediction_positions)
        emissions = logits.new_zeros(batch_size, max_words, logits.shape[-1])
        mask = torch.zeros(batch_size, max_words, dtype=torch.bool, device=logits.device)
        for b in range(batch_size):
            positions = prediction_positions[b]
            if positions:
                pos_t = torch.tensor(positions, dtype=torch.long, device=logits.device)
                emissions[b, : len(positions)] = logits[b].index_select(0, pos_t)
                mask[b, : len(positions)] = True
        return emissions, mask

    @staticmethod
    def _pad_word_labels(
        word_labels: list[list[int]],
        max_words: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pad word-level labels into a [B, max_words] long tensor."""
        batch_size = len(word_labels)
        padded = torch.zeros(batch_size, max_words, dtype=torch.long, device=device)
        for b in range(batch_size):
            labels = word_labels[b]
            if labels:
                padded[b, : len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
        return padded

    # ------------------------------------------------------------------
    # Backbone → logits helpers (shared by forward and decode)
    # ------------------------------------------------------------------

    def _get_token_hidden(self, outputs: Any) -> torch.Tensor:
        """Extract (possibly aggregated) hidden states from backbone outputs."""
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                raise ValueError("Backbone output does not expose hidden states.")
            return last_hidden
        if self.layer_weighted_sum is not None:
            selected = list(hidden_states[-self.num_aggregate_layers :])
            return self.layer_weighted_sum(selected)
        return hidden_states[-1]

    def _compute_logits(self, token_hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to label logits."""
        token_hidden = token_hidden.to(dtype=self._projector_dtype())
        token_repr = self.token_projector(token_hidden)
        if self.classifier_type == "bilinear":
            label_repr = self.entity_projector(
                self.definition_features.to(dtype=self._projector_dtype())
            )
            return self.classifier(token_repr, label_repr)
        return self.classifier(token_repr)

    # ------------------------------------------------------------------
    # Forward / decode
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        prediction_positions: list[list[int]] | None = None,
        word_labels: list[list[int]] | None = None,
    ) -> JPTModelOutput:
        """Compute logits and (optionally) loss.

        When CRF is active *and* ``prediction_positions`` / ``word_labels``
        are provided the CRF negative log-likelihood is used as the loss.
        Otherwise the standard CE / focal loss is computed from ``labels``.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        token_hidden = self._get_token_hidden(outputs)
        logits = self._compute_logits(token_hidden)

        loss = None
        if self.crf is not None and prediction_positions is not None and word_labels is not None:
            # CRF loss on word-level emissions
            word_emissions, word_mask = self._extract_word_emissions(logits, prediction_positions)
            word_label_tensor = self._pad_word_labels(
                word_labels, word_emissions.shape[1], logits.device,
            )
            loss = self.crf(word_emissions, word_label_tensor, word_mask)
        elif labels is not None:
            loss = compute_loss(
                logits=logits,
                labels=labels,
                training_config=self.training_config,
                class_weights=class_weights,
            )
        return JPTModelOutput(logits=logits, loss=loss)

    def decode_from_logits(
        self,
        logits: torch.Tensor,
        prediction_positions: list[list[int]],
    ) -> list[list[int]]:
        """Decode from pre-computed logits (avoids redundant backbone pass)."""
        with torch.no_grad():
            if self.crf is not None:
                word_emissions, word_mask = self._extract_word_emissions(logits, prediction_positions)
                return self.crf.decode(word_emissions, word_mask)
            preds = logits.argmax(dim=-1)
            return [
                [int(preds[b, pos].item()) for pos in positions]
                for b, positions in enumerate(prediction_positions)
            ]

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prediction_positions: list[list[int]],
    ) -> list[list[int]]:
        """Full forward + decode (Viterbi when CRF is active, else argmax)."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        token_hidden = self._get_token_hidden(outputs)
        logits = self._compute_logits(token_hidden)
        return self.decode_from_logits(logits, prediction_positions)

    @torch.no_grad()
    def encode_texts(
        self,
        tokenizer,
        texts: list[str],
        batch_size: int = 8,
        pooling: str = "mean",
        device: torch.device | None = None,
    ) -> torch.Tensor:
        self.backbone.eval()
        if device is None:
            device = next(self.parameters()).device
        pooled_outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = self.backbone(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                last_hidden = getattr(outputs, "last_hidden_state")
            else:
                last_hidden = hidden_states[-1]
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            if pooling == "mean":
                pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1)
            elif pooling == "last":
                lengths = attention_mask.squeeze(-1).sum(dim=1).long() - 1
                pooled = last_hidden[torch.arange(last_hidden.shape[0], device=device), lengths]
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
            pooled_outputs.append(pooled.detach().to(dtype=self._projector_dtype()).cpu())
        return torch.cat(pooled_outputs, dim=0)


def save_checkpoint(
    model: JPTNERModel,
    output_dir: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        },
        output_path / "checkpoint.pt",
    )


def load_model_from_checkpoint(
    checkpoint_dir: str | Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    num_labels: int,
    map_location: str | torch.device = "cpu",
    label_names: list[str] | None = None,
) -> tuple[JPTNERModel, dict[str, Any]]:
    model = JPTNERModel.from_config(
        model_config=model_config,
        training_config=training_config,
        num_labels=num_labels,
        label_names=label_names,
    )
    checkpoint = torch.load(Path(checkpoint_dir) / "checkpoint.pt", map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
