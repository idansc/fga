"""Metric tests, including equivalence with the original implementations."""

import math

import torch

from fga.metrics import ndcg, ranks_to_submission, scores_to_ranks, sparse_metrics


def legacy_scores_to_ranks(scores: torch.Tensor) -> torch.Tensor:
    """The original O(n * num_options) Python loop, kept as a reference oracle."""
    _, ranked_idx = scores.sort(1, descending=True)
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(scores.size(1)):
            ranks[i][ranked_idx[i][j]] = j
    return ranks + 1


def test_scores_to_ranks_matches_legacy_implementation():
    torch.manual_seed(0)
    scores = torch.randn(64, 100)
    assert torch.equal(scores_to_ranks(scores), legacy_scores_to_ranks(scores))


def test_scores_to_ranks_is_a_permutation():
    scores = torch.tensor([[0.1, 0.9, 0.5]])
    ranks = scores_to_ranks(scores)
    assert ranks.tolist() == [[3, 1, 2]]


def test_sparse_metrics_on_known_ranking():
    # Option 0 is best, option 2 second, option 1 worst.
    scores = torch.tensor([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    targets = torch.tensor([0, 2])  # ranks 1 and 2
    metrics = sparse_metrics(scores, targets)
    assert metrics["r1"] == 0.5
    assert metrics["r5"] == 1.0
    assert metrics["mean_rank"] == 1.5
    assert math.isclose(metrics["mrr"], (1.0 + 0.5) / 2)


def test_ndcg_is_one_for_a_perfect_ranking():
    relevance = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    scores = torch.tensor([[10.0, 5.0, 1.0, 0.0]])
    assert math.isclose(ndcg(scores, relevance), 1.0, rel_tol=1e-6)


def test_ndcg_penalizes_a_swapped_ranking():
    relevance = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    perfect = torch.tensor([[10.0, 5.0, 1.0, 0.0]])
    swapped = torch.tensor([[5.0, 10.0, 1.0, 0.0]])
    assert ndcg(swapped, relevance) < ndcg(perfect, relevance)


def test_ndcg_matches_hand_computation():
    # Two relevant options, model ranks the weaker one first.
    relevance = torch.tensor([[1.0, 0.5, 0.0]])
    scores = torch.tensor([[1.0, 2.0, 0.0]])
    # Model order: option 1 (rel .5), option 0 (rel 1) -> dcg = .5/1 + 1/log2(3)
    dcg = 0.5 / math.log2(2) + 1.0 / math.log2(3)
    best = 1.0 / math.log2(2) + 0.5 / math.log2(3)
    assert math.isclose(ndcg(scores, relevance), dcg / best, rel_tol=1e-6)


def test_ranks_to_submission_respects_per_image_round_counts():
    scores = torch.randn(20, 100)  # 2 images x 10 rounds
    submission = ranks_to_submission(scores, image_ids=[7, 9], num_rounds_per_image=[10, 3], num_rounds=10)
    assert len(submission) == 13
    assert {entry["image_id"] for entry in submission} == {7, 9}
    assert sorted(e["round_id"] for e in submission if e["image_id"] == 9) == [1, 2, 3]
    assert sorted(submission[0]["ranks"]) == list(range(1, 101))
