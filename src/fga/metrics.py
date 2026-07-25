"""Retrieval metrics for Visual Dialog: R@k, mean rank, MRR and NDCG.

The NDCG implementation follows the official VisDial evaluation, which scores a
single densely-annotated round per image, while the sparse metrics score every
round against its single ground-truth answer.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["scores_to_ranks", "sparse_metrics", "ndcg", "build_compute_metrics"]


def scores_to_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Convert scores to 1-based ranks, highest score getting rank 1.

    Args:
        scores: `(..., num_options)`.

    Returns: ranks of the same shape, `1` for the best-scoring option.
    """
    ranked_idx = scores.argsort(dim=-1, descending=True)
    # Inverting the permutation turns "which option sits at position j" into
    # "which position does option i sit at" -- the original code did this with a
    # Python double loop over every row and option.
    return ranked_idx.argsort(dim=-1) + 1


def sparse_metrics(scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """R@{1,5,10}, mean rank and MRR of the ground-truth answer.

    Args:
        scores: `(num_rounds, num_options)`.
        targets: `(num_rounds,)`, index of the ground-truth option.
    """
    ranks = scores_to_ranks(scores)
    gt_ranks = ranks.gather(1, targets.view(-1, 1).long()).squeeze(1).float()
    return {
        "r1": (gt_ranks <= 1).float().mean().item(),
        "r5": (gt_ranks <= 5).float().mean().item(),
        "r10": (gt_ranks <= 10).float().mean().item(),
        "mean_rank": gt_ranks.mean().item(),
        "mrr": gt_ranks.reciprocal().mean().item(),
    }


def ndcg(predicted_scores: torch.Tensor, target_relevance: torch.Tensor) -> float:
    """Normalized discounted cumulative gain against dense relevance judgements.

    Args:
        predicted_scores: `(batch, num_options)` model scores for the annotated round.
        target_relevance: `(batch, num_options)` human relevance in `[0, 1]`.
    """
    predicted_scores = predicted_scores.detach().float()
    target_relevance = target_relevance.detach().float()
    num_options = predicted_scores.size(1)

    predicted_ranks = scores_to_ranks(predicted_scores)
    # Options ordered best-first, per the model and per the annotators.
    rankings = predicted_ranks.argsort(dim=-1)
    best_rankings = target_relevance.argsort(dim=-1, descending=True)

    k = (target_relevance != 0).sum(dim=-1)
    positions = torch.arange(num_options, device=predicted_scores.device)
    discounts = torch.log2(positions.float() + 2)
    # Only the top-k positions count, where k is the number of relevant answers.
    mask = positions.unsqueeze(0) < k.unsqueeze(1)

    def dcg(order: torch.Tensor) -> torch.Tensor:
        gains = target_relevance.gather(1, order)
        return ((gains / discounts) * mask).sum(dim=-1)

    best = dcg(best_rankings)
    scores = dcg(rankings) / best.clamp(min=1e-12)
    return scores.mean().item()


def build_compute_metrics(
    dense_relevance: Optional[np.ndarray] = None,
    dense_round_ids: Optional[np.ndarray] = None,
    num_rounds: int = 10,
):
    """Build a `compute_metrics` callable for [`transformers.Trainer`].

    Args:
        dense_relevance: `(num_images, num_options)` relevance judgements, ordered
            like the images of the evaluation split. Omit to skip NDCG.
        dense_round_ids: `(num_images,)` 1-based index of the annotated round.
        num_rounds: dialog rounds per image, used to fold the flat predictions
            back into `(num_images, num_rounds, num_options)`.

    The evaluation dataset must be iterated in its natural order, which is what
    `Trainer` does (it uses a sequential sampler for evaluation).
    """

    def compute_metrics(eval_prediction) -> Dict[str, float]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        scores = torch.as_tensor(np.asarray(logits, dtype=np.float32))

        metrics: Dict[str, float] = {}
        labels = eval_prediction.label_ids
        if labels is not None:
            metrics.update(sparse_metrics(scores, torch.as_tensor(np.asarray(labels))))

        if dense_relevance is not None and dense_round_ids is not None:
            num_images = scores.size(0) // num_rounds
            per_image = scores[: num_images * num_rounds].view(num_images, num_rounds, -1)
            rounds = torch.as_tensor(np.asarray(dense_round_ids[:num_images], dtype=np.int64)) - 1
            annotated = per_image[torch.arange(num_images), rounds.clamp(0, num_rounds - 1)]
            relevance = torch.as_tensor(np.asarray(dense_relevance[:num_images], dtype=np.float32))
            metrics["ndcg"] = ndcg(annotated, relevance)

        return metrics

    return compute_metrics


def ranks_to_submission(
    scores: torch.Tensor,
    image_ids: Sequence[int],
    num_rounds_per_image: Sequence[int],
    num_rounds: int = 10,
) -> List[Dict[str, object]]:
    """Format scores as the EvalAI submission JSON.

    Args:
        scores: `(num_images * num_rounds, num_options)` in dataset order.
        image_ids: image id per image, in the same order.
        num_rounds_per_image: how many rounds each dialog actually has. The test
            split is truncated at a random round, and only that round is scored.
        num_rounds: rounds each image occupies in `scores`.
    """
    ranks = scores_to_ranks(scores).view(len(image_ids), num_rounds, -1)
    submission = []
    for i, image_id in enumerate(image_ids):
        n = int(num_rounds_per_image[i])
        for r in range(n):
            submission.append(
                {
                    "image_id": int(image_id),
                    "round_id": r + 1,
                    "ranks": ranks[i][r].tolist(),
                }
            )
    return submission
