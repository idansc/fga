"""Text-to-video retrieval, on the FGA attention layer.

A port of the matching model from https://github.com/AmeenAli/VideoMatch, which
applies Factor Graph Attention to retrieval: a text query and a video are each a
sequence — words and clips — and attention decides which clips and which words
matter *for each other* before the two are compared.

Retrieval differs from the other use cases in that there is no classifier. The two
attended representations are scored against each other, and training is
contrastive: a matching pair must outscore the hardest mismatched pair in the batch
by a margin.

The attention here deliberately leaves the entity counts unset, as the original
does, so the pairwise factors mean-marginalize their interaction grid instead of
learning the marginalization. Clip and word counts vary per example in retrieval,
and a learned marginalization would fix them.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...attention import FactorGraphAttention

__all__ = [
    "MODALITY_NAMES",
    "VideoMatchConfig",
    "VideoMatchModel",
    "VideoMatchOutput",
    "contrastive_loss",
]

#: Modality order.
MODALITY_NAMES = ("video", "text")


def contrastive_loss(similarity: torch.Tensor, margin: float = 0.2, max_violation: bool = True):
    """Bidirectional max-margin ranking loss over a batch of matched pairs.

    Args:
        similarity: `(batch, batch)` scores, where the diagonal holds the true
            video/query pairs and everything off-diagonal is a mismatch.
        margin: how far a true pair must outscore a mismatched one.
        max_violation: score against the single hardest negative rather than
            summing over all of them, which is what makes this work at scale.
    """
    positives = similarity.diag().view(-1, 1)
    cost_query = (margin + similarity - positives.expand_as(similarity)).clamp(min=0)
    cost_video = (margin + similarity - positives.t().expand_as(similarity)).clamp(min=0)

    mask = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    cost_query = cost_query.masked_fill(mask, 0)
    cost_video = cost_video.masked_fill(mask, 0)

    if max_violation:
        return cost_query.max(dim=1)[0].sum() + cost_video.max(dim=0)[0].sum()
    return cost_query.sum() + cost_video.sum()


class VideoMatchConfig(PretrainedConfig):
    r"""Configuration for [`VideoMatchModel`].

    Args:
        video_dim (`int`, *optional*, defaults to 1024): incoming clip feature size.
        text_dim (`int`, *optional*, defaults to 1024): incoming word feature size.
        hidden_size (`int`, *optional*, defaults to 1024): shared space.
        margin (`float`, *optional*, defaults to 0.2): contrastive margin.
        max_violation (`bool`, *optional*, defaults to `True`):
            Use only the hardest negative per row.
        use_context_attention (`bool`, *optional*, defaults to `True`):
            Run a second attention over the same pair to produce a context
            representation, as the original does, and average the two scores.
    """

    model_type = "video_match"

    def __init__(
        self,
        video_dim: int = 1024,
        text_dim: int = 1024,
        hidden_size: int = 1024,
        margin: float = 0.2,
        max_violation: bool = True,
        use_context_attention: bool = True,
        **kwargs,
    ):
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.hidden_size = hidden_size
        self.margin = margin
        self.max_violation = max_violation
        self.use_context_attention = use_context_attention
        super().__init__(**kwargs)


@dataclass
class VideoMatchOutput(ModelOutput):
    """Output of [`VideoMatchModel`].

    Args:
        loss (`torch.FloatTensor`, *optional*): contrastive loss, when `return_loss`.
        similarity (`torch.FloatTensor` of shape `(batch, batch)`):
            Query-to-video scores; the diagonal is the true pairs.
        video_embeds / text_embeds (`torch.FloatTensor` of shape `(batch, hidden_size)`):
            The attended, L2-normalized representations, ready for a retrieval index.
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Attention over clips and over words.
    """

    loss: Optional[torch.FloatTensor] = None
    similarity: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class VideoMatchModel(PreTrainedModel):
    """Match a text query against a video by attending over clips and words."""

    config_class = VideoMatchConfig
    base_model_prefix = "video_match"
    main_input_name = "video_features"

    def __init__(self, config: VideoMatchConfig):
        super().__init__(config)
        hidden = config.hidden_size

        self.video_projection = nn.Linear(config.video_dim, hidden)
        self.text_projection = nn.Linear(config.text_dim, hidden)

        # No entity counts: clip and word counts vary per example.
        self.attention = FactorGraphAttention(embed_dims=[hidden, hidden], modality_names=list(MODALITY_NAMES))
        self.context_attention = (
            FactorGraphAttention(embed_dims=[hidden, hidden], modality_names=list(MODALITY_NAMES))
            if config.use_context_attention
            else None
        )

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        video_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        return_loss: bool = True,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            video_features (`torch.FloatTensor` of shape `(batch, num_clips, video_dim)`):
                Encoded clips.
            text_features (`torch.FloatTensor` of shape `(batch, num_words, text_dim)`):
                Encoded query words.
            return_loss (`bool`, *optional*, defaults to `True`):
                Compute the contrastive loss, which treats the other rows of the
                batch as negatives. Turn it off for inference over a single pair.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        video = self.video_projection(video_features)
        text = self.text_projection(text_features)

        attended = self.attention(video, text, return_weights=True)
        attended, weights = attended if output_attentions else (attended[0], None)
        video_embeds, text_embeds = (F.normalize(x, dim=-1) for x in attended)

        similarity = text_embeds @ video_embeds.t()
        if self.context_attention is not None:
            context = self.context_attention(video, text)
            context_video, context_text = (F.normalize(x, dim=-1) for x in context)
            similarity = (similarity + context_text @ context_video.t()) / 2

        loss = None
        if return_loss:
            loss = contrastive_loss(similarity, self.config.margin, self.config.max_violation)

        if not return_dict:
            output = (similarity, video_embeds, text_embeds)
            output = output + ((tuple(weights),) if weights is not None else ())
            return ((loss,) + output) if loss is not None else output

        return VideoMatchOutput(
            loss=loss,
            similarity=similarity,
            video_embeds=video_embeds,
            text_embeds=text_embeds,
            attentions=tuple(weights) if weights is not None else None,
        )
