"""Answer generation for AVSD: stopping, conditioning and search.

These cover the three things perplexity cannot see. A decoder that never stops,
one whose conditioning has washed out by the fifth word, and a beam that has
collapsed to the greedy answer all train to a respectable loss and then score
badly on the metrics the task is actually ranked by.
"""

import pytest
import torch

from fga.tasks.video_dialog import AVSDForResponseGeneration, AVSDGenerationConfig
from fga.tasks.video_dialog.data import render
from fga.tasks.video_dialog.modeling_generation import HierarchicalHistoryEncoder

BATCH, VOCAB, ROUNDS, TURN_WORDS = 3, 40, 4, 6


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = AVSDGenerationConfig(
        vocab_size=VOCAB,
        max_answer_length=8,
        embed_size=16,
        question_dim=24,
        history_dim=12,
        history_hidden=12,
        hidden_size=24,
        decoder_proj=10,
        video_dim=32,
        audio_dim=16,
        num_video_streams=4,
        num_video_regions=9,
        num_audio_steps=4,
        max_question_length=5,
        dropout=0.0,
    )
    return AVSDForResponseGeneration(config).eval()


def batch(config):
    torch.manual_seed(0)
    return {
        "question_input_ids": torch.randint(1, VOCAB, (BATCH, config.max_question_length)),
        "history_input_ids": torch.randint(1, VOCAB, (BATCH, ROUNDS, TURN_WORDS)),
        "video_features": torch.randn(BATCH, config.num_video_streams, config.num_video_regions, config.video_dim),
        "audio_features": torch.randn(BATCH, config.num_audio_steps, config.audio_dim),
    }


def test_generation_stops_at_the_end_marker(model):
    """Nothing may follow `<eos>`, or the answer is longer than the model meant."""
    inputs = batch(model.config)
    with torch.no_grad():
        produced = model.generate_answers(**inputs)

    eos = model.config.eos_token_id
    for sequence in produced.tolist():
        if eos in sequence:
            after = sequence[sequence.index(eos) + 1 :]
            assert set(after) <= {0}, f"tokens after <eos>: {after}"


def test_render_drops_everything_after_the_marker():
    words = ["<eos>", "a", "b", "c"]
    assert render([2, 3, 1, 4, 4], words) == "a b"
    assert render([2, 0, 3], words) == "a b"  # padding inside is skipped, not terminal


def test_conditioning_reaches_every_decoder_position(model):
    """The question must still be an input late in the answer, not only at step one.

    Folding it into the initial hidden state alone is the failure this guards: the
    logits at the last position would then be identical for two different
    questions, because the state has been overwritten by the intervening tokens.
    """
    inputs = batch(model.config)
    answer = torch.randint(1, VOCAB, (BATCH, model.config.max_answer_length))

    with torch.no_grad():
        first = model(**inputs, answer_input_ids=answer).logits
        inputs["question_input_ids"] = torch.roll(inputs["question_input_ids"], 1, dims=0)
        second = model(**inputs, answer_input_ids=answer).logits

    last = model.config.max_answer_length - 1
    assert not torch.allclose(first[:, last], second[:, last], atol=1e-4)


def test_beam_of_width_one_is_greedy(model):
    """A one-wide beam has no choice to make, so it must reproduce greedy decoding."""
    inputs = batch(model.config)
    with torch.no_grad():
        greedy = model.generate_answers(**inputs)
        beam = model.beam_search(**inputs, beam_width=1)

    width = min(greedy.size(1), beam.size(1))
    assert torch.equal(greedy[:, :width], beam[:, :width])


def test_wider_beam_scores_at_least_as_well(model):
    """Searching more hypotheses cannot find a worse best one, up to the length penalty."""
    inputs = batch(model.config)
    with torch.no_grad():
        narrow = model.beam_search(**inputs, beam_width=1, length_penalty=0.0)
        wide = model.beam_search(**inputs, beam_width=4, length_penalty=0.0)

    def log_probability(tokens):
        answer = torch.nn.functional.pad(tokens[:, :-1], (1, 0), value=0)
        state, context, _ = model._encode(
            inputs["question_input_ids"], inputs["history_input_ids"],
            inputs["video_features"], inputs["audio_features"], False, None,
        )
        sequence, _ = model._decode(answer, context, state)
        log_probs = torch.log_softmax(model._logits(sequence), dim=-1)
        picked = log_probs.gather(2, tokens.unsqueeze(2)).squeeze(2)
        return (picked * (tokens != 0)).sum(dim=1)

    with torch.no_grad():
        assert (log_probability(wide) >= log_probability(narrow) - 1e-4).all()


def test_history_encoder_reads_turns_not_words():
    """Reordering the turns must change the encoding; flattening would hide it."""
    torch.manual_seed(0)
    encoder = HierarchicalHistoryEncoder(embed_size=8, hidden=6, out_size=5, dropout=0.0).eval()
    turns = torch.randn(BATCH, ROUNDS, TURN_WORDS, 8)

    with torch.no_grad():
        forward = encoder(turns)
        reversed_order = encoder(turns.flip(dims=[1]))

    assert forward.shape == (BATCH, 5)
    assert not torch.allclose(forward, reversed_order, atol=1e-5)
