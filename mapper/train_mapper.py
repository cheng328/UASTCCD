from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .losses import alignment_loss, balance_loss, sharpness_loss
from .node_vocab import NodeVocab, collect_node_types, iter_records
from .soft_mapper import SoftMapper, SoftMapperConfig
from .surrogate_encoder import EncoderConfig, SurrogateClassifier, SurrogateEncoder


@dataclass
class TrainConfig:
    num_categories: int = 64
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    category_dim: int = 128
    hidden_size: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 64
    max_seq_len: int = 512
    lambda_sharp: float = 0.1
    lambda_balance: float = 0.1
    lambda_align: float = 0.1
    seed: int = 42


class PairDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, object]],
        vocab: NodeVocab,
        ast_keys: Tuple[str, str],
        label_key: str,
        max_seq_len: int,
    ) -> None:
        self.records = records
        self.vocab = vocab
        self.ast_keys = ast_keys
        self.label_key = label_key
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def _nodes_to_ids(self, ast: object) -> List[int]:
        node_types = collect_node_types(ast)
        ids = [self.vocab.node_to_id[t] for t in node_types if t in self.vocab.node_to_id]
        return ids[: self.max_seq_len]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        ast1 = record.get(self.ast_keys[0])
        ast2 = record.get(self.ast_keys[1])
        seq1 = self._nodes_to_ids(ast1)
        seq2 = self._nodes_to_ids(ast2)
        label = record.get(self.label_key, 0)
        label_value = float(label) if isinstance(label, (int, float)) else 0.0
        return {"seq1": seq1, "seq2": seq2, "label": label_value}


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    seq1 = [item["seq1"] for item in batch]
    seq2 = [item["seq2"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    def pad(seqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([max(len(s), 1) for s in seqs], dtype=torch.long)
        max_len = max(lengths).item()
        out = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            if not s:
                continue
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out, lengths

    seq1_pad, len1 = pad(seq1)
    seq2_pad, len2 = pad(seq2)
    return {"seq1": seq1_pad, "len1": len1, "seq2": seq2_pad, "len2": len2, "label": labels}


def load_records(path: Path, max_records: int) -> List[Dict[str, object]]:
    records = []
    for idx, rec in enumerate(iter_records(path)):
        if max_records and idx >= max_records:
            break
        records.append(rec)
    return records


def _annealed_temperature(config: TrainConfig, epoch: int) -> float:
    if config.epochs <= 1:
        return config.temperature_end
    progress = (epoch - 1) / (config.epochs - 1)
    return config.temperature_start + progress * (config.temperature_end - config.temperature_start)


def _mapper_diagnostics(
    mapper: SoftMapper,
    node_to_id: Dict[str, int],
    seed_map: Dict[str, int],
    losses: List[float],
    config: TrainConfig,
) -> Dict[str, object]:
    with torch.no_grad():
        distributions = mapper.all_distributions()
        assignments = torch.argmax(distributions, dim=-1)
        active_categories = int(torch.unique(assignments).numel())
        eps = 1e-8
        entropy = -(distributions * torch.log(distributions + eps)).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(float(config.num_categories), device=distributions.device))
        normalized_entropy = float((entropy / max_entropy).item()) if max_entropy.item() > 0 else 0.0

        checked = 0
        matched = 0
        for node_type, cat_id in seed_map.items():
            idx = node_to_id.get(node_type)
            if idx is None or cat_id < 0 or cat_id >= config.num_categories:
                continue
            checked += 1
            matched += int(assignments[idx].item() == cat_id)
        alignment_score = matched / checked if checked else 0.0

    return {
        "K": config.num_categories,
        "temperature_start": config.temperature_start,
        "temperature_end": config.temperature_end,
        "final_temperature": mapper.config.temperature,
        "node_types": len(node_to_id),
        "active_categories": active_categories,
        "normalized_entropy": normalized_entropy,
        "alignment_score": alignment_score,
        "seed_constraints_checked": checked,
        "loss_history": losses,
    }


def train(
    data_path: Path,
    output_dir: Path,
    ast_keys: Tuple[str, str],
    label_key: str,
    seed_map: Dict[str, int],
    config: TrainConfig,
) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(data_path, max_records=0)
    vocab = NodeVocab.build(
        t for record in records for t in collect_node_types(record.get(ast_keys[0]))
    )
    vocab2 = NodeVocab.build(
        t for record in records for t in collect_node_types(record.get(ast_keys[1]))
    )
    if vocab.id_to_node != vocab2.id_to_node:
        merged = NodeVocab.build(vocab.id_to_node + vocab2.id_to_node)
    else:
        merged = vocab

    with (output_dir / "node_type_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(merged.to_json(), f, ensure_ascii=False, indent=2)

    dataset = PairDataset(records, merged, ast_keys, label_key, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)

    mapper = SoftMapper(
        SoftMapperConfig(
            num_node_types=len(merged.id_to_node),
            num_categories=config.num_categories,
            temperature=config.temperature_start,
        )
    )
    encoder = SurrogateEncoder(mapper, EncoderConfig(config.category_dim, config.hidden_size, config.dropout))
    classifier = SurrogateClassifier(encoder, config.hidden_size)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    loss_history: List[float] = []
    for epoch in range(1, config.epochs + 1):
        mapper.config.temperature = _annealed_temperature(config, epoch)
        classifier.train()
        total_loss = 0.0
        for batch in loader:
            logits = classifier(batch["seq1"], batch["len1"], batch["seq2"], batch["len2"])
            label = batch["label"]
            loss = bce(logits, label)

            distributions = mapper.all_distributions()
            loss += config.lambda_sharp * sharpness_loss(distributions)
            loss += config.lambda_balance * balance_loss(distributions)
            loss += config.lambda_align * alignment_loss(distributions, merged.node_to_id, seed_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, temperature={mapper.config.temperature:.4f}")

    torch.save({"mapper": mapper.state_dict()}, output_dir / "mapper.pt")

    diagnostics = _mapper_diagnostics(mapper, merged.node_to_id, seed_map, loss_history, config)
    with (output_dir / "mapper_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)


def load_seed_map(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, int):
            out[k] = v
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train learnable UAST mapper with surrogate encoder.")
    parser.add_argument("--input", required=True, help="Input dataset with pruned ASTs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--ast1-field", default="pruned_ast1")
    parser.add_argument("--ast2-field", default="pruned_ast2")
    parser.add_argument("--label-field", default="Label")
    parser.add_argument("--seed-map", default="", help="Optional JSON seed map")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-categories", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=None, help="Compatibility alias: set both start and end")
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-sharp", type=float, default=0.1)
    parser.add_argument("--lambda-balance", type=float, default=0.1)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    args = parser.parse_args()

    temperature_start = args.temperature if args.temperature is not None else args.temperature_start
    temperature_end = args.temperature if args.temperature is not None else args.temperature_end

    config = TrainConfig(
        num_categories=args.num_categories,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        lambda_sharp=args.lambda_sharp,
        lambda_balance=args.lambda_balance,
        lambda_align=args.lambda_align,
    )

    seed_map = load_seed_map(Path(args.seed_map)) if args.seed_map else {}

    train(
        data_path=Path(args.input),
        output_dir=Path(args.output_dir),
        ast_keys=(args.ast1_field, args.ast2_field),
        label_key=args.label_field,
        seed_map=seed_map,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

