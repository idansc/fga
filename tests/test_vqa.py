"""High-order attention for VQA — the second use case built on the same layer."""

import pytest
import torch

from fga.attention import FactorGraphAttention
from fga.attention.potentials import Ternary
from fga.tasks.vqa import (
    CompactBilinearPooling,
    HighOrderAttentionConfig,
    HighOrderAttentionForVQA,
    count_sketch,
)


@pytest.fixture
def tiny_config():
    return HighOrderAttentionConfig(
        vocab_size=200,
        num_answers=50,
        hidden_size=32,
        word_embed_dim=32,
        image_feature_dim=64,
        num_regions=12,
        max_question_length=7,
        num_choices=4,
        pooling_dim=128,
    )


@pytest.fixture
def tiny_batch(tiny_config):
    torch.manual_seed(0)
    c, batch = tiny_config, 3
    return {
        "question_input_ids": torch.randint(1, c.vocab_size, (batch, c.max_question_length)),
        "image_features": torch.randn(batch, c.num_regions, c.image_feature_dim),
        "choice_input_ids": torch.randint(1, c.num_answers, (batch, c.num_choices)),
        "labels": torch.randint(0, c.num_answers, (batch,)),
    }


# --- the ternary potential itself ---


def test_ternary_is_the_three_way_inner_product():
    """T[x,y,z] = sum_d X[x,d] Y[y,d] Z[z,d], checked against an explicit loop."""
    torch.manual_seed(0)
    ternary = Ternary(embed_size=6, x_size=4, y_size=3, z_size=2).eval()
    X, Y, Z = torch.randn(2, 4, 6), torch.randn(2, 3, 6), torch.randn(2, 2, 6)

    from fga.attention.potentials import conv1x1

    x = torch.nn.functional.normalize(conv1x1(X.transpose(1, 2), ternary.embed_X), dim=1)
    y = torch.nn.functional.normalize(conv1x1(Y.transpose(1, 2), ternary.embed_Y), dim=1)
    z = torch.nn.functional.normalize(conv1x1(Z.transpose(1, 2), ternary.embed_Z), dim=1)

    expected = torch.zeros(2, 4, 3, 2)
    for b in range(2):
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    expected[b, i, j, k] = (x[b, :, i] * y[b, :, j] * z[b, :, k]).sum()

    torch.testing.assert_close(torch.einsum("bdx,bdy,bdz->bxyz", x, y, z), expected, atol=1e-5, rtol=1e-4)


def test_ternary_returns_one_potential_per_modality():
    ternary = Ternary(embed_size=8, x_size=5, y_size=4, z_size=3).eval()
    px, py, pz = ternary(torch.randn(2, 5, 8), torch.randn(2, 4, 8), torch.randn(2, 3, 8))
    assert (px.shape, py.shape, pz.shape) == ((2, 5), (2, 4), (2, 3))


def test_ternary_gradient_reaches_all_three_modalities():
    ternary = Ternary(embed_size=8, x_size=5, y_size=4, z_size=3)
    inputs = [torch.randn(2, n, 8, requires_grad=True) for n in (5, 4, 3)]
    sum(p.sum() for p in ternary(*inputs)).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs)


# --- wiring into the attention layer ---


def test_ternary_adds_one_potential_to_each_member():
    without = FactorGraphAttention(embed_dims=[8, 8, 8], num_entities=[5, 6, 4])
    with_tri = FactorGraphAttention(embed_dims=[8, 8, 8], num_entities=[5, 6, 4], ternary_interactions=[(0, 1, 2)])
    for i in range(3):
        assert with_tri.num_of_potentials[i] == without.num_of_potentials[i] + 1


def test_ternary_actually_changes_the_output():
    """It must alter the attention, not merely add unused parameters."""
    torch.manual_seed(0)
    modalities = [torch.randn(2, n, 8) for n in (5, 6, 4)]

    torch.manual_seed(1)
    without = FactorGraphAttention(embed_dims=[8, 8, 8], num_entities=[5, 6, 4]).eval()
    torch.manual_seed(1)
    with_tri = FactorGraphAttention(
        embed_dims=[8, 8, 8], num_entities=[5, 6, 4], ternary_interactions=[(0, 1, 2)]
    ).eval()

    with torch.no_grad():
        a = without(*modalities)
        b = with_tri(*modalities)
    assert not all(torch.allclose(x, y) for x, y in zip(a, b))


def test_ternary_rejects_a_repeated_modality():
    with pytest.raises(ValueError, match="three distinct modalities"):
        FactorGraphAttention(embed_dims=[8, 8, 8], num_entities=[5, 6, 4], ternary_interactions=[(0, 0, 1)])


def test_ternary_requires_known_entity_counts():
    with pytest.raises(ValueError, match="needs num_entities"):
        FactorGraphAttention(embed_dims=[8, 8, 8], ternary_interactions=[(0, 1, 2)])


# --- compact bilinear pooling ---


def test_count_sketch_preserves_total_signed_mass():
    x = torch.randn(3, 20)
    index = torch.randint(0, 8, (20,))
    sign = torch.randint(0, 2, (20,)).float() * 2 - 1
    sketch = count_sketch(x, index, sign, 8)
    assert sketch.shape == (3, 8)
    torch.testing.assert_close(sketch.sum(-1), (x * sign).sum(-1), atol=1e-5, rtol=1e-4)


def test_cbp_approximates_the_bilinear_product():
    """The sketch preserves inner products *in expectation*.

    Compact Bilinear Pooling is a randomized sketch, so a single pair can be well
    off; the guarantee is that the estimate is unbiased and tracks the true value
    across many pairs. Asserting on one draw would be testing the seed.
    """
    torch.manual_seed(0)
    pooling = CompactBilinearPooling(32, 32, output_dim=4096, seed=0).eval()

    exact, approx = [], []
    for _ in range(64):
        x1, y1, x2, y2 = (torch.randn(1, 32) for _ in range(4))
        exact.append(torch.outer(x1[0], y1[0]).flatten() @ torch.outer(x2[0], y2[0]).flatten())
        approx.append(pooling(x1, y1)[0] @ pooling(x2, y2)[0])

    exact_t = torch.stack(exact)
    approx_t = torch.stack(approx)

    correlation = torch.corrcoef(torch.stack([exact_t, approx_t]))[0, 1]
    assert correlation > 0.9, f"correlation {correlation:.3f}"

    # Unbiased: the mean error should be small relative to the spread.
    assert (approx_t - exact_t).mean().abs() < 0.5 * exact_t.std()


def test_cbp_projections_are_saved_with_the_model(tmp_path):
    """Resampling the random projections on reload would change the function."""
    pooling = CompactBilinearPooling(16, 16, output_dim=64, seed=0)
    state = pooling.state_dict()
    assert {"x_index", "x_sign", "y_index", "y_sign"} <= set(state)

    reloaded = CompactBilinearPooling(16, 16, output_dim=64, seed=999)
    reloaded.load_state_dict(state)
    x, y = torch.randn(2, 16), torch.randn(2, 16)
    torch.testing.assert_close(pooling(x, y), reloaded(x, y))


# --- the model ---


def test_forward_scores_the_answer_vocabulary(tiny_config, tiny_batch):
    model = HighOrderAttentionForVQA(tiny_config).eval()
    with torch.no_grad():
        out = model(**tiny_batch)
    assert out.logits.shape == (3, tiny_config.num_answers)
    assert torch.isfinite(out.loss)


def test_attention_is_returned_per_modality(tiny_config, tiny_batch):
    model = HighOrderAttentionForVQA(tiny_config).eval()
    with torch.no_grad():
        out = model(**tiny_batch, output_attentions=True)
    shapes = [tuple(a.shape) for a in out.attentions]
    assert shapes == [
        (3, tiny_config.max_question_length),
        (3, tiny_config.num_regions),
        (3, tiny_config.num_choices),
    ]
    for a in out.attentions:
        torch.testing.assert_close(a.sum(-1), torch.ones(3))


def test_backward_reaches_the_ternary_factor(tiny_config, tiny_batch):
    model = HighOrderAttentionForVQA(tiny_config).train()
    model(**tiny_batch).loss.backward()
    ternary = model.attention.tri_models["tri_0_1_2"]
    assert ternary.embed_X.weight.grad is not None
    assert ternary.margin_Z.weight.grad is not None


def test_use_ternary_false_gives_the_pairwise_ablation(tiny_config, tiny_batch):
    tiny_config.use_ternary = False
    model = HighOrderAttentionForVQA(tiny_config).eval()
    assert not model.attention.ternary_interactions
    with torch.no_grad():
        assert model(**tiny_batch).logits.shape == (3, tiny_config.num_answers)


def test_save_and_from_pretrained_round_trip(tiny_config, tiny_batch, tmp_path):
    model = HighOrderAttentionForVQA(tiny_config).eval()
    with torch.no_grad():
        before = model(**tiny_batch).logits
    model.save_pretrained(tmp_path)
    reloaded = HighOrderAttentionForVQA.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(**tiny_batch).logits
    torch.testing.assert_close(before, after)
