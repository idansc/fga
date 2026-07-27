"""The three follow-up applications: video dialog, video retrieval and navigation.

Each exercises a different way of consuming the same attention — a decoder state,
a contrastive score, and a policy — so together they check that the layer is not
quietly specialized to answer ranking.
"""

import pytest
import torch

from fga.tasks.navigation import NavigationConfig, NavigationPolicy
from fga.tasks.video_dialog import AVSDConfig, AVSDEncoder
from fga.tasks.video_retrieval import VideoMatchConfig, VideoMatchModel, contrastive_loss

BATCH = 3


# --- audio-visual scene-aware dialog ---


@pytest.fixture
def avsd():
    config = AVSDConfig(
        question_dim=32,
        video_dim=64,
        audio_dim=32,
        hidden_size=32,
        num_video_streams=4,
        num_video_regions=9,
        max_question_length=5,
        num_audio_steps=4,
        history_dim=16,
    )
    return config, AVSDEncoder(config).eval()


def avsd_batch(config):
    torch.manual_seed(0)
    return {
        "question_states": torch.randn(BATCH, config.max_question_length, config.question_dim),
        "video_features": torch.randn(BATCH, config.num_video_streams, config.num_video_regions, config.video_dim),
        "audio_features": torch.randn(BATCH, config.num_audio_steps, config.audio_dim),
        "history_state": torch.randn(BATCH, config.history_dim),
        "question_lengths": torch.randint(1, config.max_question_length + 1, (BATCH,)),
    }


def test_avsd_state_concatenates_question_and_history(avsd):
    config, model = avsd
    with torch.no_grad():
        out = model(**avsd_batch(config))
    assert out.state.shape == (BATCH, config.hidden_size + config.history_dim)
    assert out.temporal_state.shape == (BATCH, config.hidden_size)


def test_avsd_attends_every_modality(avsd):
    config, model = avsd
    with torch.no_grad():
        out = model(**avsd_batch(config), output_attentions=True)
    assert len(out.attentions) == 2 + config.num_video_streams
    assert [tuple(a.shape) for a in out.attentions[1:-1]] == [(BATCH, config.num_video_regions)] * 4
    for a in out.attentions:
        torch.testing.assert_close(a.sum(-1), torch.ones(BATCH))


def test_avsd_question_prior_marks_the_final_word(avsd):
    """Only the question carries a prior, as in the original."""
    config, model = avsd
    batch = avsd_batch(config)
    with torch.no_grad():
        with_prior = model(**batch).state
        without = model(**{**batch, "question_lengths": None}).state
    assert not torch.allclose(with_prior, without)


def test_avsd_history_is_optional(avsd):
    config, model = avsd
    batch = avsd_batch(config)
    batch["history_state"] = None
    with torch.no_grad():
        assert model(**batch).state.shape == (BATCH, config.hidden_size)


# --- text-to-video retrieval ---


def test_retrieval_scores_every_query_against_every_video():
    model = VideoMatchModel(VideoMatchConfig(video_dim=32, text_dim=32, hidden_size=32)).eval()
    with torch.no_grad():
        out = model(video_features=torch.randn(BATCH, 7, 32), text_features=torch.randn(BATCH, 5, 32))
    assert out.similarity.shape == (BATCH, BATCH)
    assert out.video_embeds.shape == (BATCH, 32)
    torch.testing.assert_close(out.video_embeds.norm(dim=-1), torch.ones(BATCH))


def test_retrieval_handles_variable_length_sequences():
    """No entity counts are declared, so clip and word counts may vary."""
    model = VideoMatchModel(VideoMatchConfig(video_dim=32, text_dim=32, hidden_size=32)).eval()
    with torch.no_grad():
        for clips, words in [(4, 3), (11, 7)]:
            out = model(
                video_features=torch.randn(BATCH, clips, 32),
                text_features=torch.randn(BATCH, words, 32),
                return_loss=False,
            )
            assert out.similarity.shape == (BATCH, BATCH)


def test_contrastive_loss_is_zero_when_true_pairs_win_by_the_margin():
    similarity = torch.full((4, 4), -1.0) + torch.eye(4) * 2
    assert contrastive_loss(similarity, margin=0.2).item() == 0.0


def test_contrastive_loss_punishes_a_confused_pair():
    good = torch.full((3, 3), -1.0) + torch.eye(3) * 2
    bad = good.clone()
    bad[0, 1] = 5.0  # a mismatched pair now outscores the true one
    assert contrastive_loss(bad, margin=0.2) > contrastive_loss(good, margin=0.2)


def test_retrieval_loss_decreases_when_trained_on_one_batch():
    torch.manual_seed(0)
    model = VideoMatchModel(VideoMatchConfig(video_dim=16, text_dim=16, hidden_size=16)).train()
    video, text = torch.randn(4, 5, 16), torch.randn(4, 5, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = model(video_features=video, text_features=text).loss
    for _ in range(20):
        optimizer.zero_grad()
        model(video_features=video, text_features=text).loss.backward()
        optimizer.step()
    last = model(video_features=video, text_features=text).loss
    assert last < first


# --- target-driven navigation ---


def test_navigation_emits_a_policy_and_a_value():
    config = NavigationConfig(target_dim=16, observation_dim=32, grid_size=9, hidden_size=32, action_space=6)
    policy = NavigationPolicy(config).eval()
    with torch.no_grad():
        out = policy(target_embeds=torch.randn(BATCH, 1, 16), observation=torch.randn(BATCH, 9, 32))
    assert out.action_logits.shape == (BATCH, config.action_space)
    assert out.value.shape == (BATCH,)


def test_navigation_carries_recurrent_state_across_steps():
    """An episode is sequential: the state from one step must feed the next."""
    config = NavigationConfig(target_dim=16, observation_dim=32, grid_size=9, hidden_size=32)
    policy = NavigationPolicy(config).eval()
    target = torch.randn(BATCH, 1, 16)

    state = None
    outputs = []
    with torch.no_grad():
        for _ in range(3):
            out = policy(target_embeds=target, observation=torch.randn(BATCH, 9, 32), hidden_state=state)
            state = out.hidden_state
            outputs.append(out.action_logits)

    assert len(state) == 2 and state[0].shape == (BATCH, config.hidden_size)
    assert not torch.allclose(outputs[0], outputs[-1])


def test_navigation_attention_maps_the_observation_grid():
    config = NavigationConfig(target_dim=16, observation_dim=32, grid_size=9, hidden_size=32)
    policy = NavigationPolicy(config).eval()
    with torch.no_grad():
        out = policy(
            target_embeds=torch.randn(BATCH, 1, 16),
            observation=torch.randn(BATCH, 9, 32),
            output_attentions=True,
        )
    # target, memory, action are single vectors; the observation is the grid.
    assert [tuple(a.shape) for a in out.attentions] == [(BATCH, 1), (BATCH, 1), (BATCH, 1), (BATCH, 9)]
    torch.testing.assert_close(out.attentions[3].sum(-1), torch.ones(BATCH))


def test_navigation_gradients_reach_the_attention():
    config = NavigationConfig(target_dim=16, observation_dim=32, grid_size=9, hidden_size=32)
    policy = NavigationPolicy(config).train()
    out = policy(target_embeds=torch.randn(BATCH, 1, 16), observation=torch.randn(BATCH, 9, 32))
    (out.action_logits.sum() + out.value.sum()).backward()
    # The grid-against-target pairwise factor is the one that steers the map.
    assert policy.attention.pp_models["0_3"].embed_X.weight.grad is not None


@pytest.mark.parametrize(
    "builder",
    [
        lambda: (AVSDConfig(hidden_size=16, question_dim=16, video_dim=16, audio_dim=16), AVSDEncoder),
        lambda: (VideoMatchConfig(video_dim=16, text_dim=16, hidden_size=16), VideoMatchModel),
        lambda: (NavigationConfig(target_dim=16, observation_dim=16, hidden_size=16), NavigationPolicy),
    ],
)
def test_every_task_round_trips_through_save_pretrained(builder, tmp_path):
    config, cls = builder()
    model = cls(config)
    model.save_pretrained(tmp_path)
    reloaded = cls.from_pretrained(tmp_path)
    assert type(reloaded) is cls
    assert sum(p.numel() for p in reloaded.parameters()) == sum(p.numel() for p in model.parameters())


def test_navigation_keeps_the_spatial_map_rather_than_pooling_it():
    """The policy must know *where* the target is, not only that it is present.

    The original flattens the attention-weighted grid into the recurrent input
    (`poten * relu(state_embedding)`); pooling to one vector would discard the
    position the agent has to move toward, which is the paper's whole point.
    """
    config = NavigationConfig(target_dim=16, observation_dim=32, grid_size=9, hidden_size=32)
    policy = NavigationPolicy(config).eval()

    with torch.no_grad():
        out = policy(target_embeds=torch.randn(BATCH, 1, 16), observation=torch.randn(BATCH, 9, 32))

    assert out.attended_grid.shape == (BATCH, config.grid_size, config.attention_dim)
    # The recurrent layer consumes the whole map, and nothing else.
    assert policy.recurrent.input_size == config.grid_size * config.attention_dim


def test_navigation_attends_memory_and_the_previous_action():
    """The grid is scored against three things, not just the target.

    The original computes a similarity map per cell against the target, the
    recurrent memory *and* the last action, then mixes them. An earlier version of
    this port attended the target alone, so the agent could not use where it had
    already been or what it had just done.
    """
    config = NavigationConfig(target_dim=16, observation_dim=32, grid_size=9, hidden_size=32)
    policy = NavigationPolicy(config).eval()
    assert list(policy.attention.modality_names) == ["target", "memory", "action", "observation"]

    target = torch.randn(BATCH, 1, 16)
    observation = torch.randn(BATCH, 9, 32)
    memory = (torch.randn(BATCH, 32), torch.randn(BATCH, 32))
    action = torch.zeros(BATCH, config.action_space)
    action[:, 2] = 1.0

    with torch.no_grad():
        base = policy(target_embeds=target, observation=observation)
        with_memory = policy(target_embeds=target, observation=observation, hidden_state=memory)
        with_action = policy(target_embeds=target, observation=observation, prev_action=action)

    # Each of the two extra modalities moves the attention over the grid.
    assert not torch.allclose(base.attended_grid, with_memory.attended_grid, atol=1e-5)
    assert not torch.allclose(base.attended_grid, with_action.attended_grid, atol=1e-5)


def test_navigation_distinguishes_where_the_target_sits():
    """The same content in two different cells must give different actions."""
    torch.manual_seed(0)
    config = NavigationConfig(target_dim=16, observation_dim=16, grid_size=9, hidden_size=16)
    policy = NavigationPolicy(config).eval()

    target = torch.randn(1, 1, 16)
    base = torch.randn(1, 9, 16)

    left, right = base.clone(), base.clone()
    left[0, 0] += 3.0 * target[0, 0]
    right[0, 8] += 3.0 * target[0, 0]

    with torch.no_grad():
        a = policy(target_embeds=target, observation=left).action_logits
        b = policy(target_embeds=target, observation=right).action_logits
    assert not torch.allclose(a, b, atol=1e-4)
