from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SoftMapperConfig:
    num_node_types: int
    num_categories: int = 64
    temperature: float = 1.0


class SoftMapper(nn.Module):
    def __init__(self, config: SoftMapperConfig) -> None:
        super().__init__()
        self.config = config
        self.logits = nn.Parameter(torch.zeros(config.num_node_types, config.num_categories))

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        logits = self.logits[node_ids]
        return torch.softmax(logits / self.config.temperature, dim=-1)

    def all_distributions(self) -> torch.Tensor:
        return torch.softmax(self.logits / self.config.temperature, dim=-1)

