import pytest
import torch

from fga import FGAConfig


@pytest.fixture
def tiny_config() -> FGAConfig:
    """A small but structurally faithful config: 6 utilities, shared history factors."""
    return FGAConfig(
        vocab_size=64,
        word_embed_dim=8,
        hidden_ques_dim=12,
        hidden_ans_dim=12,
        hidden_hist_dim=6,
        hidden_cap_dim=6,
        hidden_img_dim=10,
        num_options=7,
        num_history_rounds=3,
        utility_sizes=[7, 5, 9, 4, 5, 5],
        sharing_factor_weights={4: (3, [0, 1]), 5: (3, [0, 1])},
    )


@pytest.fixture
def tiny_batch(tiny_config):
    torch.manual_seed(0)
    config = tiny_config
    batch_size = 2
    n_opt = config.num_options
    n_hist = config.num_history_rounds
    q_len, cap_len, hist_len = 5, 9, 5
    ans_len = 5
    return {
        "question_input_ids": torch.randint(1, config.vocab_size, (batch_size, q_len)),
        "option_input_ids": torch.randint(1, config.vocab_size, (batch_size, n_opt, ans_len)),
        "history_question_input_ids": torch.randint(1, config.vocab_size, (batch_size, n_hist, hist_len)),
        "history_answer_input_ids": torch.randint(1, config.vocab_size, (batch_size, n_hist, hist_len)),
        "caption_input_ids": torch.randint(1, config.vocab_size, (batch_size, cap_len)),
        "question_lengths": torch.randint(1, q_len + 1, (batch_size,)),
        "option_lengths": torch.randint(1, ans_len + 1, (batch_size, n_opt)),
        "caption_lengths": torch.randint(1, cap_len + 1, (batch_size,)),
        "image_features": torch.randn(batch_size, config.utility_sizes[3], config.hidden_img_dim),
        "labels": torch.randint(0, n_opt, (batch_size,)),
    }
