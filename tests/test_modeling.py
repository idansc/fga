"""Model tests: shapes, HuggingFace round-trips, and the documented behaviour changes."""

import pytest
import torch

from fga import FGAConfig, FGAForVisualDialog, FGAModel


def test_forward_returns_one_logit_per_option(tiny_config, tiny_batch):
    model = FGAForVisualDialog(tiny_config).eval()
    with torch.no_grad():
        out = model(**tiny_batch)
    assert out.logits.shape == (2, tiny_config.num_options)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_loss_is_omitted_without_labels(tiny_config, tiny_batch):
    model = FGAForVisualDialog(tiny_config).eval()
    tiny_batch.pop("labels")
    with torch.no_grad():
        out = model(**tiny_batch)
    assert out.loss is None


def test_backward_reaches_every_parameter(tiny_config, tiny_batch):
    model = FGAForVisualDialog(tiny_config).train()
    model(**tiny_batch).loss.backward()
    unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert unused == []


def test_save_and_from_pretrained_round_trip(tiny_config, tiny_batch, tmp_path):
    model = FGAForVisualDialog(tiny_config).eval()
    with torch.no_grad():
        before = model(**tiny_batch).logits

    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()

    reloaded = FGAForVisualDialog.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(**tiny_batch).logits
    torch.testing.assert_close(before, after)


def test_config_round_trip_keeps_integer_utility_keys(tiny_config, tmp_path):
    tiny_config.save_pretrained(tmp_path)
    reloaded = FGAConfig.from_pretrained(tmp_path)
    # json turns dict keys into strings; the config must turn them back.
    assert set(reloaded.sharing_factor_weights) == {4, 5}
    assert reloaded.sharing_factor_weights[4] == (3, [0, 1])
    assert reloaded.utility_sizes == tiny_config.utility_sizes


def test_auto_classes_resolve_fga(tiny_config, tmp_path):
    from transformers import AutoConfig, AutoModel

    FGAModel(tiny_config).save_pretrained(tmp_path)
    assert isinstance(AutoConfig.from_pretrained(tmp_path), FGAConfig)
    assert isinstance(AutoModel.from_pretrained(tmp_path), FGAModel)


def test_output_attentions_are_distributions(tiny_config, tiny_batch):
    model = FGAForVisualDialog(tiny_config).eval()
    with torch.no_grad():
        out = model(**tiny_batch, output_attentions=True)
    assert len(out.attentions) == 6
    for weights in out.attentions:
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_base_model_exposes_pooled_utilities(tiny_config, tiny_batch):
    model = FGAModel(tiny_config).eval()
    tiny_batch.pop("labels")
    with torch.no_grad():
        out = model(**tiny_batch)
    assert set(out.pooled_utilities) == {
        "answer",
        "question",
        "caption",
        "image",
        "history_question",
        "history_answer",
    }
    assert out.pooled_utilities["image"].shape == (2, tiny_config.hidden_img_dim)
    assert out.history_state.shape == (2, tiny_config.num_history_rounds * tiny_config.hidden_hist_dim)


def test_padding_embedding_row_is_zero(tiny_config):
    model = FGAForVisualDialog(tiny_config)
    embeddings = model.get_input_embeddings()
    assert torch.all(embeddings.weight[embeddings.padding_idx] == 0)


def test_eval_is_deterministic(tiny_config, tiny_batch):
    """Dropout must be off at eval; the original code left the unary one on."""
    model = FGAForVisualDialog(tiny_config).eval()
    with torch.no_grad():
        first = model(**tiny_batch).logits
        second = model(**tiny_batch).logits
    torch.testing.assert_close(first, second)


def test_lengths_beyond_the_sequence_do_not_index_out_of_bounds(tiny_config, tiny_batch):
    model = FGAForVisualDialog(tiny_config).eval()
    tiny_batch["question_lengths"] = torch.full_like(tiny_batch["question_lengths"], 999)
    tiny_batch["option_lengths"] = torch.zeros_like(tiny_batch["option_lengths"])
    with torch.no_grad():
        out = model(**tiny_batch)
    assert torch.isfinite(out.logits).all()


def test_mismatched_history_shapes_are_rejected(tiny_config, tiny_batch):
    tiny_batch["history_answer_input_ids"] = tiny_batch["history_answer_input_ids"][:, :2]
    model = FGAForVisualDialog(tiny_config).eval()
    with pytest.raises(ValueError, match="same"):
        model(**tiny_batch)


def test_runs_without_cuda(tiny_config, tiny_batch):
    """The original model hardcoded .cuda() and could not run on CPU at all."""
    model = FGAForVisualDialog(tiny_config).eval()
    assert next(model.parameters()).device.type == "cpu"
    with torch.no_grad():
        assert model(**tiny_batch).logits.device.type == "cpu"
