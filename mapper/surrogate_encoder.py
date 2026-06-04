from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .soft_mapper import SoftMapper


@dataclass
class EncoderConfig:
    category_dim: int = 128
    hidden_size: int = 256
    dropout: float = 0.1


class SurrogateEncoder(nn.Module):
    def __init__(self, mapper: SoftMapper, config: EncoderConfig) -> None:
        super().__init__()
        self.mapper = mapper
        self.config = config
        self.category_embed = nn.Linear(mapper.config.num_categories, config.category_dim, bias=False)
        self.gru = nn.GRU(
            input_size=config.category_dim,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def encode(self, node_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        distributions = self.mapper(node_ids)
        embedded = self.category_embed(distributions)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        batch_size = out.size(0)
        last_indices = (lengths - 1).clamp(min=0)
        last_vectors = out[torch.arange(batch_size), last_indices]
        return self.dropout(last_vectors)


class SurrogateClassifier(nn.Module):
    def __init__(self, encoder: SurrogateEncoder, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        rep_dim = hidden_size * 8
        self.classifier = nn.Sequential(
            nn.Linear(rep_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        seq_a: torch.Tensor,
        len_a: torch.Tensor,
        seq_b: torch.Tensor,
        len_b: torch.Tensor,
    ) -> torch.Tensor:
        h_a = self.encoder.encode(seq_a, len_a)
        h_b = self.encoder.encode(seq_b, len_b)
        rep = torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=-1)
        return self.classifier(rep).squeeze(-1)
