"""Audio-visual scene-aware dialog.

Six modalities are attended jointly: the question, four spatio-temporal video
streams, and the audio track.

What makes this use case distinct from Visual Dialog is that the video arrives as
*several* streams covering different time spans. They are attended individually,
then fused by a small LSTM over the stream axis, so the model can weigh moments
against each other after deciding what to look at within each.

The dialog history is encoded separately and concatenated with the attended
question to form the state a response decoder consumes. Only the encoder is ported
here; generation needs the AVSD dataset and a decoder of your choosing.

----

Ported from https://github.com/idansc/simple-avsd, the AVSD entry built on Factor
Graph Attention.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...attention import FactorGraphAttention

__all__ = ["AVSDConfig", "AVSDEncoder", "AVSDEncoderOutput", "MODALITY_NAMES"]


def _modality_names(num_streams: int):
    return ("question",) + tuple(f"video_{i}" for i in range(num_streams)) + ("audio",)


#: Default modality order: question, four video streams, audio.
MODALITY_NAMES = _modality_names(4)


class AVSDConfig(PretrainedConfig):
    r"""Configuration for [`AVSDEncoder`].

    Args:
        question_dim (`int`, *optional*, defaults to 256):
            Dimension of the encoded question tokens.
        video_dim (`int`, *optional*, defaults to 512):
            Dimension of the incoming video features, before projection.
        audio_dim (`int`, *optional*, defaults to 128):
            Dimension of the incoming audio features, before projection.
        hidden_size (`int`, *optional*, defaults to 256):
            Shared dimension the streams are projected to.
        num_video_streams (`int`, *optional*, defaults to 4):
            Spatio-temporal video streams, each attended separately.
        num_video_regions (`int`, *optional*, defaults to 49):
            Regions per stream, 7x7 for the conv grid used in the paper.
        max_question_length (`int`, *optional*, defaults to 10):
            Question tokens.
        num_audio_steps (`int`, *optional*, defaults to 10):
            Audio frames.
        history_dim (`int`, *optional*, defaults to 256):
            Dimension of the encoded dialog history, concatenated with the
            attended question to form the decoder state.
        use_sizes (`bool`, *optional*, defaults to `False`):
            Give the pairwise factors explicit entity counts. The original leaves
            this off (`size_flag=False`), which falls back to mean-marginalizing
            the interaction grid instead of learning the marginalization.
    """

    model_type = "avsd"

    def __init__(
        self,
        question_dim: int = 256,
        video_dim: int = 512,
        audio_dim: int = 128,
        hidden_size: int = 256,
        num_video_streams: int = 4,
        num_video_regions: int = 49,
        max_question_length: int = 10,
        num_audio_steps: int = 10,
        history_dim: int = 256,
        use_sizes: bool = False,
        **kwargs,
    ):
        self.question_dim = question_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_size = hidden_size
        self.num_video_streams = num_video_streams
        self.num_video_regions = num_video_regions
        self.max_question_length = max_question_length
        self.num_audio_steps = num_audio_steps
        self.history_dim = history_dim
        self.use_sizes = use_sizes
        super().__init__(**kwargs)

    @property
    def modality_names(self):
        return _modality_names(self.num_video_streams)


@dataclass
class AVSDEncoderOutput(ModelOutput):
    """Output of [`AVSDEncoder`].

    Args:
        state (`torch.FloatTensor` of shape `(batch, hidden_size + history_dim)`):
            Attended question concatenated with the encoded history — the state a
            response decoder is conditioned on.
        temporal_state (`torch.FloatTensor` of shape `(batch, hidden_size)`):
            The audio and video streams fused across the stream axis.
        pooled_modalities (`Dict[str, torch.FloatTensor]`, *optional*):
            The attended representation of each modality.
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Attention per modality, which is what the paper visualizes to show
            where in the video each question looks.
    """

    state: Optional[torch.FloatTensor] = None
    temporal_state: Optional[torch.FloatTensor] = None
    pooled_modalities: Optional[dict] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class AVSDEncoder(PreTrainedModel):
    """Attends a question over several video streams and the audio track."""

    config_class = AVSDConfig
    base_model_prefix = "avsd"
    main_input_name = "question_states"

    def __init__(self, config: AVSDConfig):
        super().__init__(config)
        hidden = config.hidden_size
        streams = config.num_video_streams

        self.video_projection = nn.Conv1d(config.video_dim, hidden, 1)
        self.audio_projection = nn.Conv1d(config.audio_dim, hidden, 1)

        sizes = None
        if config.use_sizes:
            sizes = [config.max_question_length] + [config.num_video_regions] * streams + [config.num_audio_steps]

        self.attention = FactorGraphAttention(
            embed_dims=[config.question_dim] + [hidden] * streams + [hidden],
            num_entities=sizes,
            modality_names=list(config.modality_names),
            use_prior=True,
        )

        # Fuse the attended audio and video streams along the stream axis, so the
        # model can compare moments after choosing where to look inside each.
        self.temporal = nn.LSTM(hidden, hidden, batch_first=True)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                nn.init.zeros_(param) if name.startswith("bias") else nn.init.kaiming_normal_(param)

    def forward(
        self,
        question_states: torch.FloatTensor,
        video_features: torch.FloatTensor,
        audio_features: torch.FloatTensor,
        history_state: Optional[torch.FloatTensor] = None,
        question_lengths: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            question_states (`torch.FloatTensor` of shape `(batch, max_question_length, question_dim)`):
                Encoded question tokens.
            video_features (`torch.FloatTensor` of shape `(batch, num_video_streams, num_video_regions, video_dim)`):
                One spatio-temporal stream per time span.
            audio_features (`torch.FloatTensor` of shape `(batch, num_audio_steps, audio_dim)`):
                Audio track features.
            history_state (`torch.FloatTensor` of shape `(batch, history_dim)`, *optional*):
                Encoded dialog history, concatenated with the attended question.
            question_lengths (`torch.LongTensor` of shape `(batch,)`, *optional*):
                Real token count per question. Used as the attention prior, which
                marks the final word — the same length cue Visual Dialog uses.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        batch, streams, regions, _ = video_features.shape

        video = self.video_projection(video_features.reshape(batch * streams, regions, -1).transpose(1, 2))
        video = video.transpose(1, 2).view(batch, streams, regions, -1)
        audio = self.audio_projection(audio_features.transpose(1, 2)).transpose(1, 2)

        # Only the question carries a prior; the paper leaves the rest uniform.
        prior = torch.zeros(batch, question_states.size(1), device=question_states.device)
        if question_lengths is not None:
            index = (question_lengths.long() - 1).clamp_(0, question_states.size(1) - 1)
            prior[torch.arange(batch, device=prior.device), index] = 1
        priors = [prior] + [None] * (streams + 1)

        modalities = [question_states, *[video[:, i] for i in range(streams)], audio]
        attended = self.attention(modalities, priors=priors, return_weights=True)
        attended, weights = attended if output_attentions else (attended[0], None)

        pooled_question, pooled_streams, pooled_audio = attended[0], attended[1:-1], attended[-1]

        # Audio first, then the streams in order, as one short sequence.
        sequence = torch.stack([pooled_audio, *pooled_streams], dim=1)
        _, (temporal_state, _) = self.temporal(sequence)
        temporal_state = temporal_state[-1]

        state = pooled_question
        if history_state is not None:
            state = torch.cat((pooled_question, history_state), dim=1)

        pooled = dict(zip(self.config.modality_names, attended))
        if not return_dict:
            output = (state, temporal_state, pooled)
            return output + ((tuple(weights),) if weights is not None else ())

        return AVSDEncoderOutput(
            state=state,
            temporal_state=temporal_state,
            pooled_modalities=pooled,
            attentions=tuple(weights) if weights is not None else None,
        )
