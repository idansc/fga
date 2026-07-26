"""The attention layer used as a standalone `nn.Module`, on tasks that are not Visual Dialog.

The point of these tests is that nothing here imports the visual-dialog package:
the layer has to stand on its own for the video-dialog, navigation and retrieval
uses the paper's follow-ups put it to.
"""

import pytest
import torch
import torch.nn as nn

from fga.attention import FactorGraphAttention, Utility


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
