"""PyTorch Factor Graph Attention model.

Factor Graph Attention (Schwartz et al., CVPR 2019) -- https://arxiv.org/abs/1904.05880
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .attention import Atten
from .configuration_fga import FGAConfig
from .encoders import build_text_encoder

__all__ = [
    "FGAModel",
    "FGAForVisualDialog",
    "FGAModelOutput",
    "FGAForVisualDialogOutput",
    "UTILITY_NAMES",
]

#: Utility order used everywhere in this file; it fixes the meaning of the
#: indices in `config.utility_sizes` and `config.sharing_factor_weights`.
UTILITY_NAMES = FGAConfig.UTILITY_NAMES


@dataclass
class FGAModelOutput(ModelOutput):
    """Output of [`FGAModel`].

    Args:
        pooled_utilities (`Dict[str, torch.FloatTensor]`):
            One attended vector per utility, keyed by [`UTILITY_NAMES`], each
            `(batch, utility_dim)`.
        answer_states (`torch.FloatTensor` of shape `(batch, num_options, hidden_ans_dim)`):
            The final LSTM state of every candidate answer, i.e. the unattended
            representation each option is scored with.
        history_state (`torch.FloatTensor` of shape `(batch, num_history_rounds * hidden_hist_dim)`):
            Attended question/answer history, fused per round and flattened.
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Attention distribution per utility, in [`UTILITY_NAMES`] order.
            Returned when `output_attentions=True`.
    """

    pooled_utilities: Optional[Dict[str, torch.FloatTensor]] = None
    answer_states: Optional[torch.FloatTensor] = None
    history_state: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class FGAForVisualDialogOutput(ModelOutput):
    """Output of [`FGAForVisualDialog`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Cross-entropy over the candidate answers, returned when `labels` is given.
        logits (`torch.FloatTensor` of shape `(batch, num_options)`):
            Score of every candidate answer. Rank them descending to get the
            submission ranking.
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            See [`FGAModelOutput`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class FGAPreTrainedModel(PreTrainedModel):
    """Handles weight initialization and the `from_pretrained` plumbing."""

    config_class = FGAConfig
    base_model_prefix = "fga"
    main_input_name = "question_input_ids"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Kaiming/Xavier init, matching `initialize_model_weights` of the paper code.

        Biases are zeroed, normalization weights are left at their default, and
        LSTM weights follow `config.lstm_initializer_type` while everything else
        follows `config.initializer_type`.
        """
        init_fns = {
            "he": nn.init.kaiming_normal_,
            "xavier": nn.init.xavier_normal_,
        }
        lstm_init = init_fns.get(self.config.lstm_initializer_type)
        other_init = init_fns.get(self.config.initializer_type)

        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if name.startswith("bias"):
                    nn.init.zeros_(param)
                elif lstm_init is not None:
                    lstm_init(param)
        elif isinstance(module, (nn.Conv1d, nn.Linear)):
            if other_init is not None:
                other_init(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if other_init is not None:
                other_init(module.weight)
            if module.padding_idx is not None:
                # The original code left this row at its random init. Because
                # `padding_idx` zeroes the gradient, it stayed random forever and
                # silently fed noise into every padded position.
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            # Left at PyTorch defaults (weight=1, bias=0), as in the paper code.
            pass


class FGAModel(FGAPreTrainedModel):
    """Factor-graph attention over the six Visual Dialog utilities.

    Returns the attended representation of each modality without scoring any
    answer, so it can be reused for other tasks (the same backbone has been
    applied to video dialog, spatial navigation and video retrieval).

    Args:
        config ([`FGAConfig`]): model configuration.
    """

    def __init__(self, config: FGAConfig):
        super().__init__(config)
        self.text_encoder = build_text_encoder(config)

        encoder_dims = self.text_encoder.output_dims
        for field, expected in (
            ("question", config.hidden_ques_dim),
            ("answer", config.hidden_ans_dim),
            ("caption", config.hidden_cap_dim),
            ("history_question", config.hidden_hist_dim),
            ("history_answer", config.hidden_hist_dim),
        ):
            if encoder_dims[field] != expected:
                raise ValueError(
                    f"Text encoder emits {encoder_dims[field]} dims for '{field}' but the config declares {expected}."
                )

        # Fuses the attended question and answer of each history round.
        self.qahistnet = nn.Sequential(
            nn.Linear(config.hidden_hist_dim * 2, config.hidden_hist_dim),
            nn.ReLU(inplace=True),
        )

        self.mul_atten = Atten(
            util_e=config.utility_dims,
            sharing_factor_weights=config.sharing_factor_weights,
            sizes=config.utility_sizes,
            use_prior=config.use_prior,
            use_pairwise=config.use_pairwise,
            use_unary=config.use_unary,
            use_self=config.use_self,
            utility_names=UTILITY_NAMES,
            size_force=config.size_force,
            unary_dropout=config.unary_dropout,
            legacy_unary_dropout=config.legacy_unary_dropout,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_encoder.set_input_embeddings(value)

    @staticmethod
    def _last_token_index(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Index of the final real token, clamped into `[0, max_len - 1]`.

        Guards against zero-length or over-long sequences, which used to index
        out of bounds and surface as an opaque device-side assert.
        """
        return (lengths.long() - 1).clamp_(0, max_len - 1)

    def forward(
        self,
        question_input_ids: torch.LongTensor,
        option_input_ids: torch.LongTensor,
        history_question_input_ids: torch.LongTensor,
        history_answer_input_ids: torch.LongTensor,
        caption_input_ids: torch.LongTensor,
        question_lengths: torch.LongTensor,
        option_lengths: torch.LongTensor,
        caption_lengths: torch.LongTensor,
        image_features: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            question_input_ids (`torch.LongTensor` of shape `(batch, question_len)`):
                Token ids of the current question.
            option_input_ids (`torch.LongTensor` of shape `(batch, num_options, answer_len)`):
                Token ids of the candidate answers.
            history_question_input_ids (`torch.LongTensor` of shape `(batch, num_history_rounds, history_len)`):
                Token ids of the questions asked in previous rounds.
            history_answer_input_ids (`torch.LongTensor` of shape `(batch, num_history_rounds, history_len)`):
                Token ids of the answers given in previous rounds.
            caption_input_ids (`torch.LongTensor` of shape `(batch, caption_len)`):
                Token ids of the image caption.
            question_lengths (`torch.LongTensor` of shape `(batch,)`):
                Number of real tokens per question; drives the attention prior.
            option_lengths (`torch.LongTensor` of shape `(batch, num_options)`):
                Number of real tokens per candidate answer.
            caption_lengths (`torch.LongTensor` of shape `(batch,)`):
                Number of real tokens per caption.
            image_features (`torch.FloatTensor` of shape `(batch, num_regions, hidden_img_dim)`):
                Pre-extracted, L2-normalized region features.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size = question_input_ids.size(0)
        num_options = option_input_ids.size(1)
        num_history_rounds, history_len = history_question_input_ids.shape[1:3]
        answer_len = option_input_ids.size(-1)
        device = question_input_ids.device

        if history_answer_input_ids.shape[1:3] != (num_history_rounds, history_len):
            raise ValueError(
                "history_question_input_ids and history_answer_input_ids must have the same "
                f"(rounds, length); got {tuple(history_question_input_ids.shape[1:3])} and "
                f"{tuple(history_answer_input_ids.shape[1:3])}."
            )

        question_states = self.text_encoder(question_input_ids, "question")
        answer_states_flat = self.text_encoder(option_input_ids.reshape(-1, answer_len), "answer")
        history_question_states = self.text_encoder(
            history_question_input_ids.reshape(-1, history_len), "history_question"
        )
        history_answer_states = self.text_encoder(history_answer_input_ids.reshape(-1, history_len), "history_answer")
        caption_states = self.text_encoder(caption_input_ids, "caption")

        # Each option is represented by its final LSTM state.
        option_last = self._last_token_index(option_lengths.reshape(-1), answer_len)
        flat_index = torch.arange(num_options * batch_size, device=device)
        answer_states = answer_states_flat[flat_index, option_last, :].view(
            batch_size, num_options, self.config.hidden_ans_dim
        )

        # Priors: spike the last word of the question/caption, stay uniform elsewhere.
        priors: List[Optional[torch.Tensor]] = [None] * 6
        if self.config.use_prior:
            batch_index = torch.arange(batch_size, device=device)
            question_prior = torch.zeros(batch_size, question_states.size(1), device=device)
            question_prior[batch_index, self._last_token_index(question_lengths, question_states.size(1))] = 100
            caption_prior = torch.zeros(batch_size, caption_states.size(1), device=device)
            caption_prior[batch_index, self._last_token_index(caption_lengths.reshape(-1), caption_states.size(1))] = (
                100
            )
            priors = [
                torch.ones(batch_size, num_options, device=device),  # answer
                question_prior,
                caption_prior,
                torch.ones(batch_size, image_features.size(1), device=device),  # image
                None,  # history question
                None,  # history answer
            ]

        utilities = [
            answer_states,
            question_states,
            caption_states,
            image_features,
            history_question_states,
            history_answer_states,
        ]

        attended = self.mul_atten(utilities, priors=priors if self.config.use_prior else None, return_weights=True)
        attended, weights = attended if output_attentions else (attended[0], None)
        (
            answer_atten,
            question_atten,
            caption_atten,
            image_atten,
            history_question_atten,
            history_answer_atten,
        ) = attended

        # Fuse the two history utilities round by round, then flatten the rounds.
        history_state = self.qahistnet(torch.cat((history_question_atten, history_answer_atten), 1))
        history_state = history_state.view(batch_size, num_history_rounds * self.config.hidden_hist_dim)

        pooled = {
            "answer": answer_atten,
            "question": question_atten,
            "caption": caption_atten,
            "image": image_atten,
            "history_question": history_question_atten,
            "history_answer": history_answer_atten,
        }

        if not return_dict:
            output = (pooled, answer_states, history_state)
            return output + (tuple(weights),) if weights is not None else output

        return FGAModelOutput(
            pooled_utilities=pooled,
            answer_states=answer_states,
            history_state=history_state,
            attentions=tuple(weights) if weights is not None else None,
        )


class FGAForVisualDialog(FGAPreTrainedModel):
    """FGA with the scoring head used for the VisDial answer-ranking task.

    Every candidate answer is scored by an MLP over its own representation
    concatenated with the attended question, answer set, image, history and
    caption. Training is a 100-way cross-entropy over the candidates.

    Example:

    ```python
    >>> from fga import FGAForVisualDialog
    >>> model = FGAForVisualDialog.from_pretrained("idansc/fga")  # doctest: +SKIP
    >>> outputs = model(**batch)  # doctest: +SKIP
    >>> ranking = outputs.logits.argsort(dim=-1, descending=True)  # doctest: +SKIP
    ```
    """

    def __init__(self, config: FGAConfig):
        super().__init__(config)
        self.fga = FGAModel(config)

        self.concat_dim = (
            config.hidden_ques_dim
            + config.hidden_ans_dim * 2
            + config.hidden_img_dim
            + config.hidden_cap_dim
            + config.hidden_hist_dim * config.num_history_rounds
        )
        self.simnet = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim // 2, bias=False),
            nn.BatchNorm1d(self.concat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.concat_dim // 2, self.concat_dim // 4, bias=False),
            nn.BatchNorm1d(self.concat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(self.concat_dim // 4, 1),
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.fga.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.fga.set_input_embeddings(value)

    def forward(
        self,
        question_input_ids: torch.LongTensor,
        option_input_ids: torch.LongTensor,
        history_question_input_ids: torch.LongTensor,
        history_answer_input_ids: torch.LongTensor,
        caption_input_ids: torch.LongTensor,
        question_lengths: torch.LongTensor,
        option_lengths: torch.LongTensor,
        caption_lengths: torch.LongTensor,
        image_features: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch,)`, *optional*):
                Index of the ground-truth answer within `option_input_ids`.
                Supply it to get the cross-entropy loss back.

        See [`FGAModel.forward`] for the remaining arguments.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.fga(
            question_input_ids=question_input_ids,
            option_input_ids=option_input_ids,
            history_question_input_ids=history_question_input_ids,
            history_answer_input_ids=history_answer_input_ids,
            caption_input_ids=caption_input_ids,
            question_lengths=question_lengths,
            option_lengths=option_lengths,
            caption_lengths=caption_lengths,
            image_features=image_features,
            output_attentions=output_attentions,
            return_dict=True,
        )

        batch_size, num_options = outputs.answer_states.shape[:2]
        pooled = outputs.pooled_utilities

        def broadcast(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unsqueeze(1).expand(batch_size, num_options, tensor.size(-1))

        features = torch.cat(
            (
                outputs.answer_states,
                broadcast(pooled["question"]),
                broadcast(pooled["answer"]),
                broadcast(pooled["image"]),
                broadcast(outputs.history_state),
                broadcast(pooled["caption"]),
            ),
            dim=2,
        )

        logits = self.simnet(features.reshape(batch_size * num_options, self.concat_dim))
        logits = logits.view(batch_size, num_options)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,) + ((outputs.attentions,) if outputs.attentions is not None else ())
            return ((loss,) + output) if loss is not None else output

        return FGAForVisualDialogOutput(loss=loss, logits=logits, attentions=outputs.attentions)
