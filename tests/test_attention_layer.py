"""The attention layer used as a standalone `nn.Module`, on tasks that are not Visual Dialog.

The point of these tests is that nothing here imports the visual-dialog package:
the layer has to stand on its own for the video-dialog, navigation and retrieval
uses the paper's follow-ups put it to.
"""

import pytest
import torch
import torch.nn as nn

from fga.attention import FactorGraphAttention, Modality, Utility


def test_used_like_any_other_nn_layer():
    attention = FactorGraphAttention(embed_dims=[512, 2048], num_entities=[20, 36])
    text = torch.randn(8, 20, 512)
    image = torch.randn(8, 36, 2048)

    pooled_text, pooled_image = attention(text, image)

    assert pooled_text.shape == (8, 512)
    assert pooled_image.shape == (8, 2048)


def test_accepts_utilities_positionally_or_as_a_sequence():
    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6]).eval()
    a, b = torch.randn(2, 5, 8), torch.randn(2, 6, 16)

    positional = attention(a, b)
    sequence = attention([a, b])

    for x, y in zip(positional, sequence):
        torch.testing.assert_close(x, y)


def test_registers_as_a_submodule_and_trains():
    """It has to behave like a layer inside a larger model, not a standalone script."""

    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6])
            self.head = nn.Linear(8 + 16, 1)

        def forward(self, a, b):
            pooled = self.attention(a, b)
            return self.head(torch.cat(pooled, dim=-1))

    model = Ranker()
    assert any(name.startswith("attention.") for name, _ in model.named_parameters())

    out = model(torch.randn(4, 5, 8), torch.randn(4, 6, 16))
    out.sum().backward()
    assert model.head.weight.grad is not None
    assert model.attention.un_models[0].embed.weight.grad is not None


def test_repr_summarises_the_graph():
    attention = FactorGraphAttention.from_utilities(
        [
            Utility("text", dim=8, size=5),
            Utility("image", dim=16, size=6),
            Utility("history", dim=8, size=5, repeats=3, connected_to=("text", "image")),
        ],
        use_prior=True,
    )
    text = repr(attention)
    assert "text:8" in text and "image:16" in text
    assert "unary+self+pairwise+prior" in text
    assert "historyx3" in text


def test_wrong_number_of_utilities_is_a_clear_error():
    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6])
    with pytest.raises(ValueError, match="built for 2 utilities"):
        attention(torch.randn(2, 5, 8))


def test_paper_argument_names_still_work():
    """Downstream forks construct it with util_e/sizes."""
    legacy = FactorGraphAttention(util_e=[8, 16], sizes=[5, 6])
    modern = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6])
    assert legacy.util_e == modern.util_e
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in modern.parameters())


def test_video_retrieval_shape():
    """Text query against video clips — the VideoMatch use case."""
    attention = FactorGraphAttention.from_utilities(
        [
            Utility("query", dim=300, size=12),
            Utility("clip", dim=1024, size=16),
        ]
    )
    pooled_query, pooled_clip = attention(torch.randn(3, 12, 300), torch.randn(3, 16, 1024))
    similarity = torch.cosine_similarity(
        nn.functional.normalize(pooled_query, dim=-1),
        nn.functional.normalize(pooled_clip[:, :300], dim=-1),
        dim=-1,
    )
    assert similarity.shape == (3,)


def test_spatial_navigation_shape():
    """An agent's observation attended against an instruction — the navigation use case."""
    attention = FactorGraphAttention.from_utilities(
        [
            Utility("instruction", dim=256, size=10),
            Utility("panorama", dim=512, size=36),
        ],
        use_prior=True,
    )
    priors = [torch.ones(2, 10), torch.ones(2, 36)]
    pooled = attention(torch.randn(2, 10, 256), torch.randn(2, 36, 512), priors=priors)
    assert [tuple(p.shape) for p in pooled] == [(2, 256), (2, 512)]


def test_three_modalities_with_a_repeated_one():
    """Video dialog: a question, the frames, and one repeated history utility."""
    attention = FactorGraphAttention.from_utilities(
        [
            Utility("question", dim=64, size=8),
            Utility("frames", dim=128, size=20),
            Utility("history", dim=32, size=8, repeats=5, connected_to=("question", "frames")),
        ]
    )
    batch, repeats = 2, 5
    pooled = attention(
        torch.randn(batch, 8, 64),
        torch.randn(batch, 20, 128),
        torch.randn(batch * repeats, 8, 32),
    )
    assert pooled[0].shape == (batch, 64)
    assert pooled[1].shape == (batch, 128)
    assert pooled[2].shape == (batch * repeats, 32)


def test_attention_weights_are_distributions():
    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6]).eval()
    _, weights = attention(torch.randn(2, 5, 8), torch.randn(2, 6, 16), return_weights=True)
    assert [tuple(w.shape) for w in weights] == [(2, 5), (2, 6)]
    for w in weights:
        torch.testing.assert_close(w.sum(-1), torch.ones(2))


def test_the_layer_does_not_import_the_task_package():
    """`fga.attention` must stay usable without the Visual Dialog code."""
    import fga.attention.factor_graph as fg
    import fga.attention.modality as mod
    import fga.attention.potentials as pot

    for module in (fg, pot, mod):
        source = module.__doc__ or ""
        assert "visual_dialog" not in getattr(module, "__file__", "")
        # no task symbols leaked into the general namespace
        assert not any(name.startswith("VisDial") or name.startswith("FGAFor") for name in dir(module)), source


# --- explicit modalities with separately declared weight sharing ---


def _shared_pair(batch=3, repeats=4):
    """The same graph expressed both ways, sharing one set of weights."""
    torch.manual_seed(0)
    compact = FactorGraphAttention.from_modalities(
        [
            Modality("answer", dim=8, size=10),
            Modality("question", dim=8, size=6),
            Modality("history", dim=4, size=5, repeats=repeats, connected_to=("answer", "question")),
        ]
    ).eval()
    torch.manual_seed(0)
    explicit = FactorGraphAttention.from_modalities(
        [Modality("answer", dim=8, size=10), Modality("question", dim=8, size=6)]
        + [Modality(f"history_{i}", dim=4, size=5, connected_to=("answer", "question")) for i in range(repeats)],
        share_weights=[[f"history_{i}" for i in range(repeats)]],
    ).eval()
    explicit.load_state_dict(compact.state_dict())
    return compact, explicit


def test_shared_weights_match_the_repeats_form_exactly():
    """The explicit spelling must be the same model, not merely a similar one."""
    batch, repeats = 3, 4
    compact, explicit = _shared_pair(batch, repeats)

    answer = torch.randn(batch, 10, 8)
    question = torch.randn(batch, 6, 8)
    history = torch.randn(batch, repeats, 5, 4)

    with torch.no_grad():
        packed = compact(answer, question, history.reshape(batch * repeats, 5, 4))
        split = explicit(answer, question, *[history[:, i] for i in range(repeats)])

    torch.testing.assert_close(packed[0], split[0])
    torch.testing.assert_close(packed[1], split[1])
    per_round = packed[2].view(batch, repeats, 4)
    for i in range(repeats):
        torch.testing.assert_close(per_round[:, i], split[2 + i])


def test_share_weights_uses_one_set_of_weights():
    compact, explicit = _shared_pair()
    assert sum(p.numel() for p in explicit.parameters()) == sum(p.numel() for p in compact.parameters())


def test_each_shared_modality_is_returned_separately():
    _, explicit = _shared_pair(batch=3, repeats=4)
    outputs = explicit(torch.randn(3, 10, 8), torch.randn(3, 6, 8), *[torch.randn(3, 5, 4) for _ in range(4)])
    assert len(outputs) == 6
    assert [tuple(o.shape) for o in outputs[2:]] == [(3, 4)] * 4


def test_share_weights_returns_per_modality_attention():
    _, explicit = _shared_pair(batch=2, repeats=4)
    _, weights = explicit(
        torch.randn(2, 10, 8),
        torch.randn(2, 6, 8),
        *[torch.randn(2, 5, 4) for _ in range(4)],
        return_weights=True,
    )
    assert len(weights) == 6
    assert [tuple(w.shape) for w in weights[2:]] == [(2, 5)] * 4
    for w in weights:
        torch.testing.assert_close(w.sum(-1), torch.ones(w.size(0)))


def test_shared_modalities_must_agree_on_shape():
    with pytest.raises(ValueError, match="match in shape"):
        FactorGraphAttention.from_modalities(
            [
                Modality("a", dim=8, size=5),
                Modality("h1", dim=4, size=5, connected_to=("a",)),
                Modality("h2", dim=6, size=5, connected_to=("a",)),
            ],
            share_weights=[["h1", "h2"]],
        )


def test_shared_modalities_must_agree_on_connections():
    with pytest.raises(ValueError, match="agree on connections"):
        FactorGraphAttention.from_modalities(
            [
                Modality("a", dim=8, size=5),
                Modality("b", dim=8, size=5),
                Modality("h1", dim=4, size=5, connected_to=("a",)),
                Modality("h2", dim=4, size=5, connected_to=("b",)),
            ],
            share_weights=[["h1", "h2"]],
        )


def test_a_modality_cannot_be_in_two_share_groups():
    with pytest.raises(ValueError, match="more than one share_weights group"):
        FactorGraphAttention.from_modalities(
            [
                Modality("a", dim=8, size=5),
                Modality("h1", dim=4, size=5, connected_to=("a",)),
                Modality("h2", dim=4, size=5, connected_to=("a",)),
                Modality("h3", dim=4, size=5, connected_to=("a",)),
            ],
            share_weights=[["h1", "h2"], ["h1", "h3"]],
        )


def test_share_weights_names_must_exist():
    with pytest.raises(ValueError, match="unknown modality"):
        FactorGraphAttention.from_modalities(
            [Modality("a", dim=8, size=5), Modality("h1", dim=4, size=5, connected_to=("a",))],
            share_weights=[["h1", "typo"]],
        )


def test_conv_era_checkpoints_migrate_to_the_linear_shape(tmp_path):
    """Old checkpoints stored the projections as Conv1d weights: (out, in, 1).

    There is deliberately no load-time shim -- `from_pretrained` bypasses module
    hooks, and hidden reshaping is the kind of magic that bites later. Instead
    the file is rewritten once by scripts/migrate_conv_checkpoint.py, and this
    test drives that script end to end.
    """
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.torch import save_file

    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6]).eval()

    legacy = {}
    for key, value in attention.state_dict().items():
        key = "mul_atten." + key
        legacy[key] = (
            value.unsqueeze(-1).contiguous() if key.endswith(".weight") and value.dim() == 2 else value.contiguous()
        )
    save_file(legacy, tmp_path / "model.safetensors", metadata={"format": "pt"})

    script = Path(__file__).parents[1] / "scripts" / "migrate_conv_checkpoint.py"
    result = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    from safetensors import safe_open

    reloaded = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6]).eval()
    with safe_open(tmp_path / "model.safetensors", framework="pt") as handle:
        state = {k[len("mul_atten.") :]: handle.get_tensor(k) for k in handle.keys()}
    reloaded.load_state_dict(state)

    a, b = torch.randn(2, 5, 8), torch.randn(2, 6, 16)
    with torch.no_grad():
        for x, y in zip(attention(a, b), reloaded(a, b)):
            torch.testing.assert_close(x, y)

    # And running it again is a no-op.
    second = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True)
    assert "nothing to do" in second.stdout


def test_masked_entities_get_no_attention():
    """Padded entities must be excluded from the softmax, not merely down-weighted."""
    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6]).eval()
    text, image = torch.randn(3, 5, 8), torch.randn(3, 6, 16)

    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[:, 3:] = False  # the last two words are padding

    with torch.no_grad():
        _, weights = attention(text, image, masks=[mask, None], return_weights=True)

    assert torch.all(weights[0][:, 3:] == 0)
    torch.testing.assert_close(weights[0].sum(1), torch.ones(3))
    # The unmasked modality is untouched.
    torch.testing.assert_close(weights[1].sum(1), torch.ones(3))


def test_pooled_vector_is_a_sum_over_real_entities_only():
    """The pooled vector is the weighted sum of the unmasked entities, nothing else."""
    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6]).eval()
    text, image = torch.randn(2, 5, 8), torch.randn(2, 6, 16)
    mask = torch.tensor([[True, True, True, False, False]] * 2)

    with torch.no_grad():
        pooled, weights = attention(text, image, masks=[mask, None], return_weights=True)

    expected = (weights[0][:, :3, None] * text[:, :3]).sum(dim=1)
    torch.testing.assert_close(pooled[0], expected)


def test_all_padding_row_stays_finite():
    """A fully padded row would softmax to NaN; it falls back to uniform."""
    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6]).eval()
    text, image = torch.randn(2, 5, 8), torch.randn(2, 6, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1] = False

    with torch.no_grad():
        pooled, weights = attention(text, image, masks=[mask, None], return_weights=True)

    assert torch.isfinite(weights[0]).all() and torch.isfinite(pooled[0]).all()
    torch.testing.assert_close(weights[0][1].sum(), torch.tensor(1.0))


def test_zero_rows_collect_exactly_no_gradient():
    """A masked entity embeds to exactly zero, and must take no gradient at all.

    Finite is not enough. `F.normalize` gives NaN here, and folding the epsilon
    under the square root gives `1/sqrt(eps)` = 1e6 -- finite, but it overflows
    bf16 downstream, and the moment a mask multiplies that `inf` by zero it is
    NaN again. A zero vector has no direction, so the scale must be zero.
    """
    from fga.attention.potentials import l2_normalize

    x = torch.zeros(2, 3, 8, requires_grad=True)
    l2_normalize(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))

    # Mixed rows: only the zero one is neutralized.
    y = torch.randn(1, 3, 8)
    y[0, 1] = 0.0
    y.requires_grad_(True)
    l2_normalize(y).sum().backward()
    torch.testing.assert_close(y.grad[0, 1], torch.zeros(8))
    assert y.grad[0, 0].abs().sum() > 0

    # And it agrees with F.normalize wherever that is well defined.
    z = torch.randn(4, 5, 16)
    torch.testing.assert_close(l2_normalize(z), torch.nn.functional.normalize(z, dim=-1))


def test_masked_attention_has_finite_gradients():
    """The end-to-end case: zeroed padded states, a mask, and a backward pass."""
    attention = FactorGraphAttention(embed_dims=[8, 16], num_entities=[5, 6])
    mask = torch.tensor([[True, True, True, False, False]] * 3)

    text = torch.randn(3, 5, 8, requires_grad=True)
    image = torch.randn(3, 6, 16, requires_grad=True)
    # Padded positions carry no state at all, as a mask-zero encoder leaves them.
    masked_text = text * mask.unsqueeze(-1)

    pooled = attention(masked_text, image, masks=[mask, None])
    (pooled[0].sum() + pooled[1].sum()).backward()

    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(image.grad).all()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name


def test_pairwise_handles_differing_dims_without_entity_counts():
    """Mean-marginalization must still respect each modality's own width.

    With `num_entities` unset the pairwise factors mean-marginalize, which is the
    path taken when entity counts vary per example. That must not change how wide
    each modality's projection is.
    """
    attention = FactorGraphAttention(embed_dims=[256, 512], num_entities=None).eval()
    text, video = torch.randn(2, 20, 256), torch.randn(2, 48, 512)

    with torch.no_grad():
        pooled_text, pooled_video = attention(text, video)

    assert pooled_text.shape == (2, 256)
    assert pooled_video.shape == (2, 512)


def test_pairwise_self_interaction_still_mirrors_x():
    """A factor with no second modality projects both sides from x's width."""
    from fga.attention.potentials import Pairwise

    factor = Pairwise(embed_x_size=64, x_spatial_dim=7, self_interaction=True)
    assert factor.embed_X.in_features == 64
    assert factor.embed_Y.in_features == 64
