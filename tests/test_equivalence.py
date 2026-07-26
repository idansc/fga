"""Prove the refactor did not change the math.

The original model is instantiated from `tests/legacy_reference/`, its weights are
saved in the old `.pth.tar` layout, converted by
`scripts/convert_legacy_checkpoint.py`, and the two models are then required to
produce identical scores for identical inputs.

Two differences have to be neutralized to make the comparison deterministic:

* the original called `F.dropout` in the unary potential without forwarding
  `self.training`, so it dropped activations even under `.eval()`. It is patched
  out here and the new model is configured with `unary_dropout=0`.
* the original hardcoded `.cuda()` in several places. `Tensor.cuda` is patched to
  a no-op so the reference can run on CPU at all.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fga import FGAConfig, FGAForVisualDialog

sys.path.insert(0, str(Path(__file__).parent))

from legacy_reference import legacy_atten  # noqa: E402
from legacy_reference.legacy_fga_model import FGA as LegacyFGA  # noqa: E402

VOCAB_SIZE = 40
WORD_DIM = 8
QUES_DIM = 12
ANS_DIM = 12
HIST_DIM = 6
CAP_DIM = 6
IMG_DIM = 10


@pytest.fixture
def patched_legacy(monkeypatch):
    """Make the reference implementation runnable and deterministic on CPU."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(legacy_atten.F, "dropout", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(F, "dropout", lambda x, *args, **kwargs: x)
    return True


def build_legacy_model():
    torch.manual_seed(1234)
    # The legacy constructor adds 1+1 rows internally, so pass the word count.
    model = LegacyFGA(
        vocab_size=VOCAB_SIZE - 2,
        word_embed_dim=WORD_DIM,
        hidden_ques_dim=QUES_DIM,
        hidden_ans_dim=ANS_DIM,
        hidden_hist_dim=HIST_DIM,
        hidden_cap_dim=CAP_DIM,
        hidden_img_dim=IMG_DIM,
    )
    return model.eval()


def legacy_inputs(batch_size=2):
    torch.manual_seed(7)
    n_options = 100
    return {
        "input_ques": torch.randint(1, VOCAB_SIZE, (batch_size, 21)),
        "input_ans": torch.randint(1, VOCAB_SIZE, (batch_size, n_options, 21)),
        "input_hist_ques": torch.randint(1, VOCAB_SIZE, (batch_size, 9, 21)),
        "input_hist_ans": torch.randint(1, VOCAB_SIZE, (batch_size, 9, 21)),
        "input_hist_cap": torch.randint(1, VOCAB_SIZE, (batch_size, 41)),
        "input_ques_length": torch.randint(1, 22, (batch_size,)),
        "input_ans_length": torch.randint(1, 22, (batch_size, n_options)),
        "input_cap_length": torch.randint(1, 42, (batch_size,)),
        "i_e": torch.randn(batch_size, 37, IMG_DIM),
    }


def to_new_inputs(legacy):
    return {
        "question_input_ids": legacy["input_ques"],
        "option_input_ids": legacy["input_ans"],
        "history_question_input_ids": legacy["input_hist_ques"],
        "history_answer_input_ids": legacy["input_hist_ans"],
        "caption_input_ids": legacy["input_hist_cap"],
        "question_lengths": legacy["input_ques_length"],
        "option_lengths": legacy["input_ans_length"],
        "caption_lengths": legacy["input_cap_length"],
        "image_features": legacy["i_e"],
    }


def new_config():
    return FGAConfig(
        vocab_size=VOCAB_SIZE,
        word_embed_dim=WORD_DIM,
        hidden_ques_dim=QUES_DIM,
        hidden_ans_dim=ANS_DIM,
        hidden_hist_dim=HIST_DIM,
        hidden_cap_dim=CAP_DIM,
        hidden_img_dim=IMG_DIM,
        num_history_rounds=9,
        utility_sizes=[100, 21, 41, 37, 21, 21],
        sharing_factor_weights={4: (9, [0, 1]), 5: (9, [0, 1])},
        unary_dropout=0.0,
        classifier_dropout=0.5,
    )


def test_parameter_counts_match_except_dead_weights(patched_legacy):
    legacy = build_legacy_model()
    new = FGAForVisualDialog(new_config())
    legacy_params = sum(p.numel() for p in legacy.parameters())
    new_params = sum(p.numel() for p in new.parameters())
    # The only removed weights are the six unused self-interaction margin_Y convs.
    dead = sum(
        module.margin_Y.weight.numel() + module.margin_Y.bias.numel()
        for name, module in legacy.mul_atten.pp_models.items()
        if "(" not in name
    )
    assert new_params == legacy_params - dead


def test_converted_checkpoint_reproduces_legacy_scores(patched_legacy, tmp_path):
    legacy = build_legacy_model()
    checkpoint = tmp_path / "best_model_mrr.pth.tar"
    torch.save({"model": legacy.state_dict(), "epoch": 3, "metrics": {"mrr": 0.66}}, checkpoint)

    output_dir = tmp_path / "converted"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "convert_legacy_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(output_dir),
            "--visdial_params",
            "/nonexistent.json",
            "--image_feature_dim",
            str(IMG_DIM),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    converted = FGAForVisualDialog.from_pretrained(output_dir)
    converted.config.unary_dropout = 0.0
    converted = FGAForVisualDialog.from_pretrained(output_dir, config=converted.config).eval()

    inputs = legacy_inputs()
    with torch.no_grad():
        expected = legacy(**inputs)
        actual = converted(**to_new_inputs(inputs)).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_attention_module_matches_legacy_attention(patched_legacy):
    """The reusable Atten block on its own, independent of the VisDial wiring."""
    from fga.attention import Atten

    torch.manual_seed(99)
    util_e = [4, 5, 6]
    sizes = [7, 8, 9]
    # The legacy signature defaults `sharing_factor_weights` to `[]` and then calls
    # `.items()` on it, so it has to be handed a dict explicitly (see
    # `test_legacy_attention_cannot_be_used_without_sharing_weights`).
    legacy = legacy_atten.Atten(
        util_e=util_e, sizes=sizes, prior_flag=True, pairwise_flag=True, sharing_factor_weights={}
    ).eval()
    new = Atten(util_e=util_e, sizes=sizes, prior_flag=True, pairwise_flag=True, unary_dropout=0.0).eval()

    renamed = {}
    for key, value in legacy.state_dict().items():
        if key.startswith("pp_models."):
            rest = key[len("pp_models.") :]
            head, tail = rest.split(".", 1)
            head = f"{head[1:-1].replace(', ', '_')}" if head.startswith("(") else f"self_{head}"
            key = f"pp_models.{head}.{tail}"
        # The oracle's projections are Conv1d(kernel_size=1): same numbers with a
        # trailing singleton axis the Linear modules do not have.
        if key.endswith(".weight") and value.dim() == 3 and value.size(-1) == 1:
            value = value.squeeze(-1)
        renamed[key] = value
    missing, unexpected = new.load_state_dict(renamed, strict=False)
    assert not missing
    assert all("margin_Y" in key for key in unexpected)

    utils = [torch.randn(3, size, dim) for size, dim in zip(sizes, util_e)]
    priors = [torch.randn(3, size) for size in sizes]
    with torch.no_grad():
        expected = legacy([u.clone() for u in utils], priors=[p.clone() for p in priors])
        actual = new([u.clone() for u in utils], priors=[p.clone() for p in priors])

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_legacy_attention_cannot_be_used_without_sharing_weights(patched_legacy):
    """Regression: the legacy default was `sharing_factor_weights=[]`, a list.

    Any standalone use of the attention block -- the reusable part of the paper --
    therefore crashed with `'list' object has no attribute 'items'`. The current
    signature defaults to `None` and normalizes to an empty dict.
    """
    from fga.attention import Atten

    with pytest.raises(AttributeError, match="'list' object has no attribute 'items'"):
        legacy_atten.Atten(util_e=[4, 5], sizes=[6, 7])

    block = Atten(util_e=[4, 5], sizes=[6, 7])
    outputs = block([torch.randn(2, 6, 4), torch.randn(2, 7, 5)])
    assert [tuple(o.shape) for o in outputs] == [(2, 4), (2, 5)]
