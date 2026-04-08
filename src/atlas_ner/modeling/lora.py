"""Minimal LoRA adapters for frozen backbones."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoraStats:
    replaced_modules: int
    trainable_params: int


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": torch.float32,
        }
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False, **factory_kwargs)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        lora_input = self.dropout(x).to(dtype=self.lora_a.weight.dtype)
        update = self.lora_b(self.lora_a(lora_input)) * self.scaling
        return base + update.to(dtype=base.dtype)


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_adapters(
    module: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> LoraStats:
    freeze_module(module)
    replace_names = [
        name
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear) and any(name.endswith(target) for target in target_modules)
    ]
    for module_name in replace_names:
        parent, child_name = _get_parent_module(module, module_name)
        original = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout))

    trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return LoraStats(replaced_modules=len(replace_names), trainable_params=trainable_params)
