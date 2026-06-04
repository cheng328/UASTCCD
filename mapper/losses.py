from __future__ import annotations

from typing import Dict, Optional

import torch


def sharpness_loss(distributions: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    entropy = -(distributions * torch.log(distributions + eps)).sum(dim=-1)
    return entropy.mean()


def balance_loss(distributions: torch.Tensor) -> torch.Tensor:
    mean_usage = distributions.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    return torch.mean((mean_usage - target) ** 2)


def alignment_loss(
    distributions: torch.Tensor,
    node_type_ids: Dict[str, int],
    seed_map: Optional[Dict[str, int]],
) -> torch.Tensor:
    if not seed_map:
        return torch.tensor(0.0, device=distributions.device)

    losses = []
    for node_type, cat_id in seed_map.items():
        idx = node_type_ids.get(node_type)
        if idx is None or cat_id < 0 or cat_id >= distributions.size(1):
            continue
        target = torch.full((distributions.size(1),), 0.0, device=distributions.device)
        target[cat_id] = 1.0
        losses.append(torch.nn.functional.mse_loss(distributions[idx], target))

    if not losses:
        return torch.tensor(0.0, device=distributions.device)
    return torch.stack(losses).mean()

