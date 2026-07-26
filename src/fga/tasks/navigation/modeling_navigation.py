"""Target-driven visual navigation.

An agent is told an object to find and must act from egocentric observations.

Two modalities are attended jointly: the word embedding of the target object and
the spatial grid of the current observation.

The attention here is used differently from every other task in this package. The
others pool each modality to a single vector; navigation must keep the *map*. The
attention re-weights each grid cell and the whole weighted grid is flattened into
the recurrent state, so the policy sees both what was found and where it is —
pooling would discard the position the agent needs to move toward. The original
writes this as `poten * relu(state_embedding)` flattened into the LSTM input.

This use case differs from the others in what it does with the attention. There is
no ranking or classification over candidates: the output is a policy, trained by
reinforcement learning against navigation episodes rather than by a supervised
loss. Only the network is ported here — the environment, episodes and A3C training
loop live in the original repository.

----

Ported from https://github.com/barmayo/spatial_attention.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...attention import FactorGraphAttention

__all__ = ["MODALITY_NAMES", "NavigationConfig", "NavigationPolicy", "NavigationOutput"]

#: Modality order.
MODALITY_NAMES = ("target", "observation")


class NavigationConfig(PretrainedConfig):
    r"""Configuration for [`NavigationPolicy`].

    Args:
        target_dim (`int`, *optional*, defaults to 300):
            Target-object embedding size, GloVe-sized in the original.
        observation_dim (`int`, *optional*, defaults to 512):
            Channel count of the observation grid, e.g. a ResNet layer.
        grid_size (`int`, *optional*, defaults to 49):
            Spatial cells in the observation, 7x7 for a ResNet conv grid.
        num_target_tokens (`int`, *optional*, defaults to 1):
            Tokens describing the target. A single embedding for one object noun.
        hidden_size (`int`, *optional*, defaults to 512):
            Shared attention dimension and recurrent state size.
        action_space (`int`, *optional*, defaults to 6):
            Discrete actions, e.g. move/rotate/look/done.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout on the fused state. Policies are usually trained without it.
    """

    model_type = "fga_navigation"

    def __init__(
        self,
        target_dim: int = 300,
        observation_dim: int = 512,
        grid_size: int = 49,
        num_target_tokens: int = 1,
        hidden_size: int = 512,
        action_space: int = 6,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.target_dim = target_dim
        self.observation_dim = observation_dim
        self.grid_size = grid_size
        self.num_target_tokens = num_target_tokens
        self.hidden_size = hidden_size
        self.action_space = action_space
        self.dropout = dropout
        super().__init__(**kwargs)


@dataclass
class NavigationOutput(ModelOutput):
    """Output of [`NavigationPolicy`].

    Args:
        action_logits (`torch.FloatTensor` of shape `(batch, action_space)`):
            Unnormalized action scores.
        value (`torch.FloatTensor` of shape `(batch,)`):
            State-value estimate, for the critic.
        hidden_state (`Tuple[torch.FloatTensor, torch.FloatTensor]`):
            Recurrent state to carry into the next step of the episode.
        attended_grid (`torch.FloatTensor` of shape `(batch, grid_size, hidden_size)`):
            The observation with each cell scaled by its attention — the spatial
            map the policy acts on, before flattening.
        pooled_modalities (`Dict[str, torch.FloatTensor]`, *optional*)
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Attention over the target tokens and over the spatial grid — the
            latter being the map of where the agent is looking, which is the
            point of the method.
    """

    action_logits: Optional[torch.FloatTensor] = None
    value: Optional[torch.FloatTensor] = None
    attended_grid: Optional[torch.FloatTensor] = None
    hidden_state: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None
    pooled_modalities: Optional[dict] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class NavigationPolicy(PreTrainedModel):
    """Attend a navigation target over an egocentric observation, then act.

    Example:

    ```python
    >>> from fga.tasks.navigation import NavigationConfig, NavigationPolicy
    >>> policy = NavigationPolicy(NavigationConfig())
    >>> state = None
    >>> for observation, target in episode:  # doctest: +SKIP
    ...     out = policy(target_embeds=target, observation=observation, hidden_state=state)
    ...     state = out.hidden_state
    ```
    """

    config_class = NavigationConfig
    base_model_prefix = "navigation"
    main_input_name = "observation"

    def __init__(self, config: NavigationConfig):
        super().__init__(config)
        hidden = config.hidden_size

        self.target_projection = nn.Linear(config.target_dim, hidden)
        self.observation_projection = nn.Conv1d(config.observation_dim, hidden, 1)

        self.attention = FactorGraphAttention(
            embed_dims=[hidden, hidden],
            num_entities=[config.num_target_tokens, config.grid_size],
            modality_names=list(MODALITY_NAMES),
        )

        # The episode is sequential, so the attended map feeds a recurrent state.
        # Its input is the flattened grid, not a pooled vector: the policy has to
        # know where the target is, not only that it is present.
        self.recurrent = nn.LSTMCell(config.grid_size * hidden + hidden, hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.actor = nn.Linear(hidden, config.action_space)
        self.critic = nn.Linear(hidden, 1)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTMCell):
            for name, param in module.named_parameters():
                nn.init.zeros_(param) if name.startswith("bias") else nn.init.kaiming_normal_(param)

    def forward(
        self,
        target_embeds: torch.FloatTensor,
        observation: torch.FloatTensor,
        hidden_state: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            target_embeds (`torch.FloatTensor` of shape `(batch, num_target_tokens, target_dim)`):
                Embedding of the object the agent is looking for.
            observation (`torch.FloatTensor` of shape `(batch, grid_size, observation_dim)`):
                Egocentric visual features, flattened over the spatial grid.
            hidden_state (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Recurrent state from the previous step; zeros at episode start.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        target = self.target_projection(target_embeds)
        grid = self.observation_projection(observation.transpose(1, 2)).transpose(1, 2)

        attended, weights = self.attention(target, grid, return_weights=True)
        pooled_target, pooled_observation = attended

        # Scale each cell by its attention and keep the map; only the target,
        # which has no spatial extent, is pooled.
        attended_grid = weights[1].unsqueeze(-1) * grid
        fused = torch.cat((attended_grid.flatten(1), pooled_target), dim=-1)
        hidden_state = self.recurrent(fused, hidden_state)
        state = self.dropout(hidden_state[0])

        action_logits = self.actor(state)
        value = self.critic(state).squeeze(-1)

        pooled = dict(zip(MODALITY_NAMES, attended))
        if not return_dict:
            output = (action_logits, value, hidden_state, attended_grid, pooled)
            return output + ((tuple(weights),) if output_attentions else ())

        return NavigationOutput(
            action_logits=action_logits,
            value=value,
            hidden_state=hidden_state,
            attended_grid=attended_grid,
            pooled_modalities=pooled,
            attentions=tuple(weights) if output_attentions else None,
        )
