#!/usr/bin/env python
"""Evaluate an ensemble of FGA checkpoints on VisDial val.

The paper reports 5xFGA alongside the single model, and the 2020 challenge
submission (https://github.com/idansc/mrr-ndcg) turns on how the members are
combined. Two combination rules are supported, because they are not equivalent:

* `--combine score` averages the raw scores. Simple, and the natural choice when
  the members are the same architecture trained from different seeds, since their
  scores are on a comparable scale.
* `--combine rank` averages the *ranks* instead (Borda count). Scale-free, so it
  is the safer choice when members disagree in confidence -- notably when mixing
  an MRR-tuned model with a dense-finetuned one, whose score distributions differ
  sharply after the soft-label objective reshapes them.

Scores for every member are computed once and cached in memory, so adding a
combination rule costs nothing extra.

```bash
python scripts/ensemble_eval.py \
    --models models/fga-frcnn/checkpoint-48160 models/fga-seed1/checkpoint-X ... \
    --image_features_path data/frcnn_features_new.h5 --combine score
```
"""

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga import FGAForVisualDialog, VisDialCollator, VisDialDataset  # noqa: E402
from fga.tasks.visual_dialog.data import (  # noqa: E402
    image_ids_from_params,
    load_visdial_params,
    vocab_size_from_params,
)
from fga.tasks.visual_dialog.metrics import ndcg, scores_to_ranks, sparse_metrics  # noqa: E402
from fga.tasks.visual_dialog.trainer import load_dense_annotations  # noqa: E402


@torch.no_grad()
def score_split(model_path: str, dataset: VisDialDataset, batch_size: int, device: str) -> np.ndarray:
    """Every candidate's score for every round, in dataset order."""
    model = FGAForVisualDialog.from_pretrained(model_path).to(device).eval()
    collate = VisDialCollator()

    scores = np.empty((len(dataset), model.config.num_options), dtype=np.float32)
    for start in range(0, len(dataset), batch_size):
        batch = collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        scores[start : start + logits.size(0)] = logits.float().cpu().numpy()

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def combine(member_scores: List[np.ndarray], rule: str) -> torch.Tensor:
    """Fuse member scores into one ranking signal."""
    tensors = [torch.from_numpy(s) for s in member_scores]
    if rule == "score":
        return torch.stack(tensors).mean(dim=0)
    if rule == "rank":
        # Lower rank is better, so negate to keep "higher is better".
        ranks = torch.stack([scores_to_ranks(t).float() for t in tensors])
        return -ranks.mean(dim=0)
    raise ValueError(f"unknown combine rule {rule!r}")


def evaluate(scores: torch.Tensor, labels: np.ndarray, dense, num_rounds: int) -> dict:
    metrics = sparse_metrics(scores, torch.from_numpy(labels))
    num_images = scores.size(0) // num_rounds
    per_image = scores[: num_images * num_rounds].view(num_images, num_rounds, -1)
    rounds = torch.from_numpy(dense["round_ids"][:num_images].astype(np.int64)) - 1
    annotated = per_image[torch.arange(num_images), rounds.clamp(0, num_rounds - 1)]
    metrics["ndcg"] = ndcg(annotated, torch.from_numpy(dense["relevance"][:num_images]))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="Checkpoint directories to ensemble.")
    parser.add_argument("--image_features_path", default="data/frcnn_features_new.h5")
    parser.add_argument("--visdial_data_path", default="data/visdial_data.h5")
    parser.add_argument("--visdial_params_path", default="data/visdial_params.json")
    parser.add_argument("--dense_annotations_path", default="data/visdial_1.0_val_dense_annotations.json")
    parser.add_argument("--combine", nargs="+", default=["score", "rank"], help="score | rank")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    params = load_visdial_params(args.visdial_params_path)
    dataset = VisDialDataset(
        visdial_data_path=args.visdial_data_path,
        image_features_path=args.image_features_path,
        split="val",
        vocab_size=vocab_size_from_params(params),
        in_memory=False,
    )
    dense = load_dense_annotations(args.dense_annotations_path, image_ids_from_params(params, "val"))
    labels = dataset.ans_index.astype(np.int64)

    results = {}
    member_scores = []
    for path in args.models:
        print(f"scoring {path}")
        scores = score_split(path, dataset, args.batch_size, args.device)
        member_scores.append(scores)
        name = os.path.basename(path.rstrip("/")) or path
        results[f"member:{name}"] = evaluate(torch.from_numpy(scores), labels, dense, dataset.n_qa_per_dial)

    for rule in args.combine:
        results[f"ensemble:{rule} (n={len(member_scores)})"] = evaluate(
            combine(member_scores, rule), labels, dense, dataset.n_qa_per_dial
        )

    header = f"{'':<42} {'NDCG':>7} {'MRR':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRank':>7}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<42} {m['ndcg'] * 100:7.2f} {m['mrr'] * 100:7.2f} {m['r1'] * 100:7.2f} "
            f"{m['r5'] * 100:7.2f} {m['r10'] * 100:7.2f} {m['mean_rank']:7.2f}"
        )

    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
