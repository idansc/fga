"""Dense-relevance objectives, which optimize NDCG rather than the sparse label."""

import math

import torch

from fga.tasks.visual_dialog.dense import approx_ndcg_loss, dense_soft_cross_entropy
from fga.tasks.visual_dialog.metrics import ndcg

RELEVANCE = torch.tensor([[1.0, 0.5, 0.0, 0.0, 0.8]])
PERFECT = torch.tensor([[10.0, 5.0, 0.0, 0.0, 7.0]])
BAD = torch.tensor([[0.0, 1.0, 9.0, 8.0, 0.5]])


def test_approx_ndcg_prefers_the_better_ranking():
    """Regression: summing the pairwise sigmoid over the wrong axis inverts this."""
    assert ndcg(PERFECT, RELEVANCE) > ndcg(BAD, RELEVANCE)
    assert approx_ndcg_loss(PERFECT, RELEVANCE) < approx_ndcg_loss(BAD, RELEVANCE)


def test_approx_ndcg_converges_on_the_true_metric_as_it_sharpens():
    loss = float(approx_ndcg_loss(PERFECT, RELEVANCE, temperature=0.05))
    assert math.isclose(1.0 - loss, ndcg(PERFECT, RELEVANCE), abs_tol=1e-3)


def test_soft_cross_entropy_prefers_the_better_ranking():
    assert dense_soft_cross_entropy(PERFECT, RELEVANCE) < dense_soft_cross_entropy(BAD, RELEVANCE)


def test_both_losses_ignore_rows_with_no_relevance():
    """An all-zero row carries no signal; a uniform target would flatten the ranking."""
    zeros = torch.zeros(1, 5)
    assert float(approx_ndcg_loss(PERFECT, zeros)) == 0.0
    assert float(dense_soft_cross_entropy(PERFECT, zeros)) == 0.0


def test_losses_are_differentiable():
    for loss_fn in (approx_ndcg_loss, dense_soft_cross_entropy):
        scores = PERFECT.clone().requires_grad_(True)
        loss_fn(scores, RELEVANCE).backward()
        assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_gradient_descends_toward_the_better_ranking():
    """Optimizing the surrogate must improve the true NDCG, not just itself.

    Adam rather than SGD, deliberately: ApproxNDCG's gradient is far smaller in
    magnitude than the cross entropy's, because NDCG gains are bounded by 1 and
    the position discount is flat. A scale-invariant optimizer makes the two
    comparable, and is what training actually uses.
    """
    for loss_fn in (approx_ndcg_loss, dense_soft_cross_entropy):
        scores = BAD.clone().requires_grad_(True)
        before = ndcg(scores.detach(), RELEVANCE)
        optimizer = torch.optim.Adam([scores], lr=0.5)
        for _ in range(100):
            optimizer.zero_grad()
            loss_fn(scores, RELEVANCE).backward()
            optimizer.step()
        after = ndcg(scores.detach(), RELEVANCE)
        assert after > before, f"{loss_fn.__name__}: {before:.4f} -> {after:.4f}"
