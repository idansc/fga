"""Answer generation for Audio-Visual Scene-Aware Dialog.

[`AVSDEncoder`] attends the question over the video and audio streams and returns
the state a decoder is conditioned on; the task itself is to *write* the answer,
scored by BLEU, METEOR, ROUGE-L and CIDEr against the reference. This adds the
decoder and the token embeddings around that encoder.

The question, the history and the answer each get their own embedding table by
default, which is what the original release does -- its `embed_model` is `None`,
so every module builds its own. Tying them is a defensible idea (same language,
same video, and one table to learn instead of three) and `tie_embeddings` turns it
on, but it is a change to the model, not a tidying of it: three tables of
`vocab x 128` are about 1.5M parameters, a fifth of the total, and the paper's
reported 8,359,107 only reconciles with three.

Reading alongside the original release: its variables are terse, and the names
here are the same quantities spelled out.

| original | here | what it is |
| --- | --- | --- |
| `ei` | `encoded.pooled_modalities` | one attended vector per modality |
| `a_q`, `a_s`, `a_a` | `pooled["question"]`, `pooled["video_*"]`, `pooled["audio"]` | the question, stream and audio halves of it |
| `es` | `dialog_context` | attended question + encoded history, shown at every step |
| `hidden_temporal_state` | `video_state` | fused streams, as the decoder's initial `(hidden, cell)` |
| `hxc` | inside [`AVSDForResponseGeneration._decode`] | word embeddings with the context concatenated on |
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

    Defaults are the released configuration: word embeddings of 128 feeding a
    single-layer question LSTM of 256, a hierarchical history encoder ending at
    128, and a 256-wide decoder projected through 128 before the vocabulary. That
    comes to about 8.36M parameters, which is what the paper reports.

    Args:
        vocab_size (`int`, *optional*, defaults to 6055): shared word vocabulary,
            including the padding id 0 and the end marker id 1.
        max_answer_length (`int`, *optional*, defaults to 20): tokens generated.
        decoder_layers (`int`, *optional*, defaults to 1): LSTM decoder depth.
        eos_token_id (`int`, *optional*, defaults to 1): the end-of-answer marker.
            The decoder is trained to emit it and stops there when generating.
        embed_size (`int`, *optional*, defaults to 128): width of the shared word
            embedding table.
        question_layers (`int`, *optional*, defaults to 1): depth of the LSTM that
            turns question word embeddings into the states attention runs over.
            Attending over raw embeddings instead costs the word order.
        history_hidden (`int`, *optional*, defaults to 128): width of the
            per-turn LSTM in the history encoder.
        history_word_layers (`int`, *optional*, defaults to 2): depth of that
            per-turn LSTM.
        history_pair_layers (`int`, *optional*, defaults to 1): depth of the LSTM
            that reads the encoded turns in order.
        decoder_proj (`int`, *optional*, defaults to 128): width of the projection
            between the decoder and the vocabulary.
        dropout (`float`, *optional*, defaults to 0.5): applied inside the LSTMs
            and before the output projection.
    """

    model_type = "fga_avsd_generation"

    def __init__(
        self,
        vocab_size: int = 6055,
        max_answer_length: int = 20,
        decoder_layers: int = 1,
        eos_token_id: int = 1,
        embed_size: int = 128,
        question_layers: int = 1,
        history_hidden: int = 128,
        history_word_layers: int = 2,
        history_pair_layers: int = 1,
        decoder_proj: int = 128,
        dropout: float = 0.5,
        tie_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_answer_length = max_answer_length
        self.decoder_layers = decoder_layers
        self.embed_size = embed_size
        self.question_layers = question_layers
        self.history_hidden = history_hidden
        self.history_word_layers = history_word_layers
        self.history_pair_layers = history_pair_layers
        self.decoder_proj = decoder_proj
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        kwargs.setdefault("pad_token_id", 0)
        super().__init__(eos_token_id=eos_token_id, **kwargs)


class HierarchicalHistoryEncoder(nn.Module):
    """Encode the dialog so far as turns, not as one long sentence.

    A flat encoder over the concatenated words has to rediscover where each turn
    began, and truncating that sequence cuts words rather than turns. Reading each
    question-answer pair on its own and then reading the pairs in order keeps the
    boundary the dialog actually has, and makes "the previous turn" a position in
    a short sequence rather than sixty words back.

    Shape:
        - Input: `(batch, rounds, words, embed_size)`
        - Output: `(batch, out_size)`
    """

    def __init__(self, embed_size, hidden, out_size, word_layers=2, pair_layers=1, dropout=0.5):
        super().__init__()
        self.per_turn = nn.LSTM(
            embed_size, hidden, num_layers=word_layers, batch_first=True,
            dropout=dropout if word_layers > 1 else 0.0,
        )
        self.across_turns = nn.LSTM(
            hidden, out_size, num_layers=pair_layers, batch_first=True,
            dropout=dropout if pair_layers > 1 else 0.0,
        )

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        batch, rounds, words, width = embedded.shape
        _, (turn_state, _) = self.per_turn(embedded.reshape(batch * rounds, words, width))
        turns = turn_state[-1].view(batch, rounds, -1)
        _, (dialog_state, _) = self.across_turns(turns)
        return dialog_state[-1]


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
        def embedding_table():
            return nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)

        # One table per consumer unless tied; see the module docstring.
        self.embedding = embedding_table()
        self.question_embedding = self.embedding if config.tie_embeddings else embedding_table()
        self.history_embedding = self.embedding if config.tie_embeddings else embedding_table()
        # Attention runs over LSTM states, not over the embeddings themselves:
        # what a word contributes depends on the words before it, and a lookup
        # table cannot express that.
        self.question_encoder = nn.LSTM(
            config.embed_size, config.question_dim, num_layers=config.question_layers,
            batch_first=True, dropout=config.dropout if config.question_layers > 1 else 0.0,
        )
        self.history_encoder = HierarchicalHistoryEncoder(
            config.embed_size, config.history_hidden, config.history_dim,
            config.history_word_layers, config.history_pair_layers, config.dropout,
        )
        self.encoder = AVSDEncoder(config)

        # Conditioning reaches the decoder along two paths, and which one carries
        # what is the whole design.
        #
        # `dialog_context` -- the attended question with the encoded history
        # appended -- is concatenated onto *every* input token, so the decoder
        # cannot stop seeing what was asked. Folding it into the initial hidden
        # state instead looks equivalent and is not: an LSTM rewrites its state at
        # every step, so within five or six words the question has washed out and
        # the decoder is running as a plain language model over the answer
        # vocabulary. It then writes fluent, video-shaped sentences that answer
        # nothing -- and swapping the video features underneath changes almost
        # nothing, because almost nothing is reading them.
        #
        # The initial state is what the *video* is for: the fused streams arrive
        # as the temporal LSTM's own (hidden, cell). The question keeps its own
        # width here, since attention pools each modality in that modality's space.
        self.dialog_context_size = config.question_dim + config.history_dim
        self.decoder = nn.LSTM(
            config.embed_size + self.dialog_context_size,
            config.hidden_size,
            num_layers=config.decoder_layers,
            batch_first=True,
            dropout=config.dropout if config.decoder_layers > 1 else 0.0,
        )
        # A narrow projection before the vocabulary, as in the original. The
        # output layer is the largest matrix in the model, so its input width is
        # where most of the parameters are decided.
        self.projection = nn.Linear(config.hidden_size, config.decoder_proj)
        self.output = nn.Linear(config.decoder_proj, config.vocab_size)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """He everywhere except the recurrent weights, which get Xavier.

        This is `initialize_model_weights(model, "he", "xavier")` in the original:
        two separate schemes, and the split matters. He assumes a ReLU, which
        halves the variance of what passes through it; an LSTM's gates are
        sigmoid and tanh, so nothing is being halved and He is roughly twice the
        right scale. For the decoder here that is a standard deviation of 0.072
        against Xavier's 0.038, which does not diverge -- it just spends the early
        epochs recovering.

        It runs over the whole model, the nested [`AVSDEncoder`] included, so the
        `Conv1d` stream projections are listed too: left out, they would keep
        PyTorch's default and quietly break the setting being reproduced.
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.1)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, parameter in module.named_parameters():
                nn.init.zeros_(parameter) if name.startswith("bias") else nn.init.xavier_normal_(parameter)

    def _encode(
        self, question_input_ids, history_input_ids, video_features, audio_features, output_attentions,
        frame_features=None,
    ):
        """Everything the decoder needs, as two named pieces.

        Returns `(video_state, dialog_context, encoded)`:

        * `video_state` -- the fused audio and video streams as an LSTM
          `(hidden, cell)` pair, ready to be the decoder's starting point.
        * `dialog_context` -- `(batch, dialog_context_size)`, the attended
          question and the encoded history, to be repeated across answer
          positions.
        * `encoded` -- the full encoder output, kept for attention maps.
        """
        question, _ = self.question_encoder(self.question_embedding(question_input_ids))
        history = self.history_encoder(self.history_embedding(history_input_ids))
        encoded = self.encoder(
            question_states=question,
            video_features=video_features,
            audio_features=audio_features,
            frame_features=frame_features,
            history_state=history,
            output_attentions=output_attentions,
        )
        layers = self.config.decoder_layers
        video_state = (
            encoded.temporal_state.unsqueeze(0).repeat(layers, 1, 1).contiguous(),
            encoded.temporal_cell.unsqueeze(0).repeat(layers, 1, 1).contiguous(),
        )
        return video_state, encoded.state, encoded

    def _decode(self, answer_tokens, dialog_context, state):
        """Run the decoder over `answer_tokens`, re-showing the context at each.

        Each position reads its own word embedding with `dialog_context`
        concatenated on, so the question is an input at every step rather than a
        memory the state has to preserve.
        """
        embedded = self.embedding(answer_tokens)
        repeated_context = dialog_context.unsqueeze(1).expand(-1, embedded.size(1), -1)
        return self.decoder(torch.cat((embedded, repeated_context), dim=-1), state)

    def forward(
        self,
        question_input_ids: torch.LongTensor,
        history_input_ids: torch.LongTensor,
        video_features: torch.FloatTensor,
        audio_features: torch.FloatTensor,
        answer_input_ids: Optional[torch.LongTensor] = None,
        frame_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        video_state, dialog_context, encoded = self._encode(
            question_input_ids, history_input_ids, video_features, audio_features, output_attentions,
            frame_features,
        )

        loss = None
        logits = None
        if answer_input_ids is not None:
            # Teacher forcing: the decoder reads the reference shifted right and
            # predicts the next token, so position t never sees token t.
            shifted = F.pad(answer_input_ids[:, :-1], (1, 0), value=0)
            sequence, _ = self._decode(shifted, dialog_context, video_state)
            logits = self._logits(sequence)
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
    def generate_answers(
        self, question_input_ids, history_input_ids, video_features, audio_features, frame_features=None
    ):
        """Greedy decoding, returning token ids `(batch, max_answer_length)`."""
        state, dialog_context, _ = self._encode(
            question_input_ids, history_input_ids, video_features, audio_features, False, frame_features
        )

        token = torch.zeros(len(dialog_context), 1, dtype=torch.long, device=dialog_context.device)
        # Once a sequence emits `<eos>` everything after it is padding. Without
        # this every answer runs to `max_answer_length`, which perplexity never
        # notices -- the surplus tokens are not scored -- while BLEU and CIDEr
        # punish the length directly.
        finished = torch.zeros(len(dialog_context), 1, dtype=torch.bool, device=dialog_context.device)
        produced = []
        for _ in range(self.config.max_answer_length):
            sequence, state = self._decode(token, dialog_context, state)
            token = self._logits(sequence).argmax(-1)
            token = token.masked_fill(finished, 0)
            produced.append(token)
            finished = finished | (token == self.config.eos_token_id)
            if bool(finished.all()):
                break
        return torch.cat(produced, dim=1)

    def _logits(self, sequence):
        return self.output(F.dropout(self.projection(sequence), self.config.dropout, self.training))

    @torch.no_grad()
    def beam_search(
        self, question_input_ids, history_input_ids, video_features, audio_features,
        frame_features=None, beam_width: int = 3, length_penalty: float = 1.0,
    ):
        """Beam search, returning token ids `(batch, <= max_answer_length)`.

        Greedy decoding commits to a first word before knowing what follows, which
        for answers this short is most of the sentence. The paper measures the
        difference: BLEU-4 goes from 0.082 without beam search to 0.096 with it,
        the largest decoding effect in its table.

        `length_penalty` divides a hypothesis's log-probability by its length
        raised to this power, so stopping early is not rewarded merely for
        accumulating fewer negative terms.
        """
        state, dialog_context, _ = self._encode(
            question_input_ids, history_input_ids, video_features, audio_features, False, frame_features
        )
        batch, device = len(dialog_context), dialog_context.device
        eos, width = self.config.eos_token_id, beam_width

        # Each example carries `width` hypotheses, laid out example-major so a
        # view of `(batch, width, ...)` always groups an example's own beams.
        context = dialog_context.repeat_interleave(width, dim=0)
        state = tuple(s.repeat_interleave(width, dim=1) for s in state)
        token = torch.zeros(batch * width, 1, dtype=torch.long, device=device)
        tokens = torch.zeros(batch * width, 0, dtype=torch.long, device=device)

        # Only the first beam of each example starts alive. Otherwise the first
        # step expands `width` identical hypotheses and the beam collapses to
        # `width` copies of the greedy answer.
        scores = torch.full((batch, width), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.reshape(-1)
        done = torch.zeros(batch * width, dtype=torch.bool, device=device)

        for _ in range(self.config.max_answer_length):
            sequence, next_state = self._decode(token, context, state)
            log_probs = F.log_softmax(self._logits(sequence).squeeze(1).float(), dim=-1)

            # A finished hypothesis may only extend itself with padding, at no cost.
            log_probs[done] = float("-inf")
            log_probs[done, 0] = 0.0

            vocab = log_probs.size(-1)
            scores, flat = (scores.unsqueeze(1) + log_probs).view(batch, -1).topk(width, dim=-1)
            beam = (flat // vocab + torch.arange(batch, device=device).unsqueeze(1) * width).reshape(-1)
            token = (flat % vocab).reshape(-1, 1)

            tokens = torch.cat((tokens[beam], token), dim=1)
            state = tuple(s[:, beam] for s in next_state)
            scores = scores.reshape(-1)
            done = done[beam] | (token.squeeze(1) == eos)
            if bool(done.all()):
                break

        lengths = (tokens != 0).sum(dim=1).clamp_min(1)
        ranked = (scores / lengths.to(scores) ** length_penalty).view(batch, width)
        best = ranked.argmax(dim=1) + torch.arange(batch, device=device) * width
        return tokens[best]
