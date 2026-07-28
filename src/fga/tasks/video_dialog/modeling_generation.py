"""Answer generation for Audio-Visual Scene-Aware Dialog.

[`AVSDEncoder`] attends the question over the video and audio streams and returns
the state a decoder is conditioned on; the task itself is to *write* the answer,
scored by BLEU, METEOR, ROUGE-L and CIDEr against the reference. This adds the
decoder and the token embeddings around that encoder.

The question, the history and the answer share one embedding table. They are the
same language about the same video, and tying them means the decoder starts from a
representation the encoder has already shaped rather than learning the vocabulary
twice.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .modeling_avsd import AVSDConfig, AVSDEncoder

__all__ = ["AVSDGenerationConfig", "AVSDForResponseGeneration", "AVSDGenerationOutput"]


class AVSDGenerationConfig(AVSDConfig):
    r"""Configuration for [`AVSDForResponseGeneration`].

    Adds to [`AVSDConfig`]:

    Args:
        vocab_size (`int`, *optional*, defaults to 6055): shared word vocabulary,
            including the padding id 0.
        max_answer_length (`int`, *optional*, defaults to 20): tokens generated.
        decoder_layers (`int`, *optional*, defaults to 1): LSTM decoder depth.
    """

    model_type = "fga_avsd_generation"

    def __init__(self, vocab_size: int = 6055, max_answer_length: int = 20, decoder_layers: int = 1, **kwargs):
        self.vocab_size = vocab_size
        self.max_answer_length = max_answer_length
        self.decoder_layers = decoder_layers
        super().__init__(**kwargs)


@dataclass
class AVSDGenerationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class AVSDForResponseGeneration(PreTrainedModel):
    """Answer a question about a video, in words.

    ```python
    >>> model = AVSDForResponseGeneration(AVSDGenerationConfig())
    >>> out = model(question_input_ids=q, history_input_ids=h,
    ...             video_features=v, audio_features=a, answer_input_ids=y)  # doctest: +SKIP
    ```
    """

    config_class = AVSDGenerationConfig
    base_model_prefix = "avsd"
    main_input_name = "question_input_ids"

    def __init__(self, config: AVSDGenerationConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.question_dim, padding_idx=0)
        self.history_encoder = nn.GRU(config.question_dim, config.history_dim, batch_first=True)
        self.encoder = AVSDEncoder(config)

        # The decoder is conditioned on the attended state and the fused streams,
        # projected to its initial hidden state. The attended question keeps its
        # own width -- attention pools each modality in that modality's space, so
        # this is question_dim, not hidden_size.
        conditioning = config.question_dim + config.history_dim + config.hidden_size
        self.to_hidden = nn.Linear(conditioning, config.hidden_size)
        self.decoder = nn.LSTM(
            config.question_dim, config.hidden_size, num_layers=config.decoder_layers, batch_first=True
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def _encode(self, question_input_ids, history_input_ids, video_features, audio_features, output_attentions):
        question = self.embedding(question_input_ids)
        _, history = self.history_encoder(self.embedding(history_input_ids))
        encoded = self.encoder(
            question_states=question,
            video_features=video_features,
            audio_features=audio_features,
            history_state=history.squeeze(0),
            output_attentions=output_attentions,
        )
        conditioning = torch.cat((encoded.state, encoded.temporal_state), dim=-1)
        return self.to_hidden(conditioning), encoded

    def forward(
        self,
        question_input_ids: torch.LongTensor,
        history_input_ids: torch.LongTensor,
        video_features: torch.FloatTensor,
        audio_features: torch.FloatTensor,
        answer_input_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        hidden, encoded = self._encode(
            question_input_ids, history_input_ids, video_features, audio_features, output_attentions
        )

        layers = self.config.decoder_layers
        state = (hidden.unsqueeze(0).repeat(layers, 1, 1).contiguous(),
                 torch.zeros_like(hidden).unsqueeze(0).repeat(layers, 1, 1).contiguous())

        loss = None
        logits = None
        if answer_input_ids is not None:
            # Teacher forcing: the decoder reads the reference shifted right and
            # predicts the next token, so position t never sees token t.
            inputs = F.pad(answer_input_ids[:, :-1], (1, 0), value=0)
            sequence, _ = self.decoder(self.embedding(inputs), state)
            logits = self.output(sequence)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                answer_input_ids.reshape(-1),
                ignore_index=0,  # padding is not a target
            )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return AVSDGenerationOutput(
            loss=loss,
            logits=logits,
            attentions=encoded.attentions,
        )

    @torch.no_grad()
    def generate_answers(self, question_input_ids, history_input_ids, video_features, audio_features):
        """Greedy decoding, returning token ids `(batch, max_answer_length)`."""
        hidden, _ = self._encode(question_input_ids, history_input_ids, video_features, audio_features, False)
        layers = self.config.decoder_layers
        state = (hidden.unsqueeze(0).repeat(layers, 1, 1).contiguous(),
                 torch.zeros_like(hidden).unsqueeze(0).repeat(layers, 1, 1).contiguous())

        token = torch.zeros(len(hidden), 1, dtype=torch.long, device=hidden.device)
        produced = []
        for _ in range(self.config.max_answer_length):
            sequence, state = self.decoder(self.embedding(token), state)
            token = self.output(sequence).argmax(-1)
            produced.append(token)
        return torch.cat(produced, dim=1)
