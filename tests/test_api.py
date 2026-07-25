"""The readable API surface: named utilities, `use_*` flags, and their aliases."""

import pytest
import torch

from fga import FGAConfig
from fga.attention import Atten, Utility


def visdial_utilities():
    return [
        Utility("answer", dim=12, size=100),
        Utility("question", dim=12, size=21),
        Utility("caption", dim=6, size=41),
        Utility("image", dim=10, size=37),
        Utility("history_question", dim=6, size=21, repeats=9, connected_to=("answer", "question")),
        Utility("history_answer", dim=6, size=21, repeats=9, connected_to=("answer", "question")),
    ]


def test_named_utilities_build_the_same_graph_as_indices():
    named = Atten.from_utilities(visdial_utilities(), use_prior=True)
    indexed = Atten(
        util_e=[12, 12, 6, 10, 6, 6],
        sizes=[100, 21, 41, 37, 21, 21],
        sharing_factor_weights={4: (9, [0, 1]), 5: (9, [0, 1])},
        use_prior=True,
    )
    assert named.sharing_factor_weights == indexed.sharing_factor_weights
    assert named.num_of_potentials == indexed.num_of_potentials
    assert sorted(named.pp_models) == sorted(indexed.pp_models)
    assert sum(p.numel() for p in named.parameters()) == sum(p.numel() for p in indexed.parameters())


def test_named_utilities_label_the_module():
    attention = Atten.from_utilities(visdial_utilities(), use_prior=True)
    assert attention.utility_names[3] == "image"
    assert "image" in attention.describe()
    assert "shared x9 connected to [answer, question]" in attention.describe()


def test_flag_aliases_still_work():
    """Downstream forks pass the paper's `*_flag` names."""
    legacy_style = Atten(util_e=[4, 5], sizes=[6, 7], prior_flag=True, pairwise_flag=False)
    assert legacy_style.use_prior is True
    assert legacy_style.use_pairwise is False
    # and the old attribute names still read back
    assert legacy_style.prior_flag is True
    assert legacy_style.pairwise_flag is False


def test_shared_utility_without_connections_is_rejected():
    with pytest.raises(ValueError, match="connected_to"):
        Utility("history", dim=6, size=21, repeats=9)


def test_connections_between_two_shared_utilities_are_rejected():
    with pytest.raises(ValueError, match="not supported"):
        Atten.from_utilities(
            [
                Utility("answer", dim=4, size=5),
                Utility("hist_q", dim=4, size=5, repeats=3, connected_to=("hist_a",)),
                Utility("hist_a", dim=4, size=5, repeats=3, connected_to=("answer",)),
            ]
        )


def test_unknown_connection_names_the_offender():
    with pytest.raises(ValueError, match="typo"):
        Atten.from_utilities(
            [
                Utility("answer", dim=4, size=5),
                Utility("history", dim=4, size=5, repeats=3, connected_to=("typo",)),
            ]
        )


def test_duplicate_utility_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        Atten.from_utilities([Utility("a", dim=4, size=5), Utility("a", dim=4, size=5)])


def test_config_accepts_the_readable_shared_form():
    config = FGAConfig(
        shared_utilities=[
            {"name": "history_question", "repeats": 9, "connected_to": ["answer", "question"]},
            {"name": "history_answer", "repeats": 9, "connected_to": ["answer", "question"]},
        ]
    )
    assert config.sharing_factor_weights == {4: (9, [0, 1]), 5: (9, [0, 1])}


def test_config_renders_shared_utilities_by_name():
    config = FGAConfig()
    assert config.shared_utilities == [
        {"name": "history_question", "repeats": 9, "connected_to": ["answer", "question"]},
        {"name": "history_answer", "repeats": 9, "connected_to": ["answer", "question"]},
    ]


def test_config_rejects_both_spellings_at_once():
    with pytest.raises(ValueError, match="not both"):
        FGAConfig(
            sharing_factor_weights={4: (9, [0, 1])},
            shared_utilities=[{"name": "history_answer", "repeats": 9, "connected_to": ["answer"]}],
        )


def test_config_rejects_an_unknown_utility_name():
    with pytest.raises(ValueError, match="Unknown utility"):
        FGAConfig(shared_utilities=[{"name": "nonsense", "repeats": 2, "connected_to": ["answer"]}])


def test_named_utilities_run_a_forward_pass():
    attention = Atten.from_utilities(
        [
            Utility("text", dim=4, size=6),
            Utility("image", dim=8, size=5),
            Utility("history", dim=4, size=6, repeats=3, connected_to=("text", "image")),
        ]
    )
    outputs = attention(
        [torch.randn(2, 6, 4), torch.randn(2, 5, 8), torch.randn(2 * 3, 6, 4)],
    )
    assert [tuple(o.shape) for o in outputs] == [(2, 4), (2, 8), (6, 4)]
