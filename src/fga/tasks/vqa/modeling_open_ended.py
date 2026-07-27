"""Open-ended VQA: answer from a fixed vocabulary, with no candidate answers given.

The multiple-choice model in [`modeling_hoa`] takes the candidate answers as a
third modality, which is what lets it use a ternary factor over
(region, word, answer) triples. Open-ended VQA has no candidates: the model sees
only the question and the image, and picks from the answer vocabulary directly.

That has two consequences worth stating, because they are easy to mistake for a
weaker port:

* With two modalities there is no ternary factor to apply. Attention is unary,
  self and one pairwise interaction. The high-order machinery is not disabled
  here; it simply has nothing to act on until a third modality exists.
* The answer can no longer influence where the model looks. In multiple choice,
  a candidate answer steers attention toward the regions that would confirm it;
  open-ended attention is driven by the question alone.

Everything else — encoders, pooling, classifier — is shared with the
multiple-choice model.

----

The multiple-choice model this shares its encoders with is ported from
https://github.com/idansc/HighOrderAtten.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...attention import FactorGraphAttention
from .layers import GatedTanh
from .modeling_hoa import QuestionEncoder
from .pooling import CompactBilinearPooling, signed_sqrt

__all__ = ["OpenEndedVQAConfig", "OpenEndedVQAModel", "OpenEndedVQAOutput", "OPEN_ENDED_MODALITIES"]

#: Modality order.
OPEN_ENDED_MODALITIES = ("question", "image")


class OpenEndedVQAConfig(PretrainedConfig):
    r"""Configuration for [`OpenEndedVQAModel`].

    Args:
        vocab_size (`int`, *optional*, defaults to 12604): question vocabulary.
        num_answers (`int`, *optional*, defaults to 3000):
            Answer vocabulary. VQA is conventionally treated as classification
            over the most frequent answers, which covers most of the data.
        hidden_size (`int`, *optional*, defaults to 512): shared dimension.
        word_embed_dim (`int`, *optional*, defaults to 512): word embedding size.
        image_feature_dim (`int`, *optional*, defaults to 2048): region feature size.
        num_regions (`int`, *optional*, defaults to 196): image regions, 14x14.
        max_question_length (`int`, *optional*, defaults to 15): question words.
        pooling_dim (`int`, *optional*, defaults to 16000): CBP sketch size.
        loss_type (`str`, *optional*, defaults to `"bce"`):
            How the answer is supervised.

            - `"bce"` — sigmoid outputs and binary cross entropy against the VQA
              score every answer earns. VQA is graded and often has several
              acceptable answers, so it is a multi-label regression, not a
              single-label classification. This is what the 2017 challenge
              winners identified as the single most useful change.
            - `"soft_ce"` — softmax cross entropy against the same scores
              renormalized to a distribution. Keeps the answers competing.
            - `"ce"` — cross entropy against one label, the original's objective.

            `"bce"` and `"soft_ce"` need `answer_scores`, which the dataset
            supplies when given `num_answers`.
        gated_tanh (`bool`, *optional*, defaults to `True`):
            Use [`GatedTanh`] in the image encoder instead of a plain `tanh`.
        soft_targets (`bool`, *optional*, defaults to `None`):
            Deprecated. `True` means `loss_type="soft_ce"`.
        dropout (`float`, *optional*, defaults to 0.5): encoder dropout.
        classifier_dropout (`float`, *optional*, defaults to 0.3): head dropout.
        mask_padding (`bool`, *optional*, defaults to `True`):
            Keep attention off padded question words, and zero the encoder state
            there. See [`HighOrderAttentionConfig`].
    """

    model_type = "open_ended_vqa"

    def __init__(
        self,
        vocab_size: int = 12604,
        num_answers: int = 3000,
        hidden_size: int = 512,
        word_embed_dim: int = 512,
        image_feature_dim: int = 2048,
        num_regions: int = 196,
        max_question_length: int = 15,
        pooling_dim: int = 16000,
        loss_type: str = "bce",
        gated_tanh: bool = True,
        soft_targets: Optional[bool] = None,
        dropout: float = 0.5,
        classifier_dropout: float = 0.3,
        mask_padding: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.num_answers = num_answers
        self.hidden_size = hidden_size
        self.word_embed_dim = word_embed_dim
        self.image_feature_dim = image_feature_dim
        self.num_regions = num_regions
        self.max_question_length = max_question_length
        self.pooling_dim = pooling_dim
        if soft_targets is not None:
            loss_type = "soft_ce" if soft_targets else "ce"
        if loss_type not in ("bce", "soft_ce", "ce"):
            raise ValueError(f"loss_type must be one of bce, soft_ce, ce; got {loss_type!r}.")
        self.loss_type = loss_type
        self.gated_tanh = gated_tanh
        self.dropout = dropout
        self.classifier_dropout = classifier_dropout
        self.mask_padding = mask_padding
        kwargs.setdefault("pad_token_id", 0)
        super().__init__(**kwargs)


@dataclass
class OpenEndedVQAOutput(ModelOutput):
    """Output of [`OpenEndedVQAModel`].

    Args:
        loss (`torch.FloatTensor`, *optional*): cross entropy over the answer vocabulary.
        logits (`torch.FloatTensor` of shape `(batch, num_answers)`): answer scores.
        pooled_modalities (`Dict[str, torch.FloatTensor]`, *optional*)
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Attention over question words and image regions.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    pooled_modalities: Optional[dict] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class OpenEndedVQAModel(PreTrainedModel):
    """Answer a question about an image by choosing from the answer vocabulary.

    Example:

    ```python
    >>> from fga.tasks.vqa import OpenEndedVQAConfig, OpenEndedVQAModel
    >>> model = OpenEndedVQAModel(OpenEndedVQAConfig())
    >>> out = model(question_input_ids=question, image_features=regions)  # doctest: +SKIP
    >>> answer_id = out.logits.argmax(-1)  # doctest: +SKIP
    ```
    """

    config_class = OpenEndedVQAConfig
    base_model_prefix = "open_ended_vqa"
    main_input_name = "question_input_ids"

    def __init__(self, config: OpenEndedVQAConfig):
        super().__init__(config)
        hidden = config.hidden_size

        self.question_encoder = QuestionEncoder(config)
        self.image_encoder = nn.Sequential(
            GatedTanh(config.image_feature_dim, hidden)
            if config.gated_tanh
            else nn.Sequential(nn.Linear(config.image_feature_dim, hidden), nn.Tanh()),
            nn.Dropout(config.dropout),
        )

        # Two modalities, so unary + self + one pairwise; a ternary factor needs
        # a third and there is none without candidate answers.
        self.attention = FactorGraphAttention(
            embed_dims=[hidden, hidden],
            num_entities=[config.max_question_length, config.num_regions],
            modality_names=list(OPEN_ENDED_MODALITIES),
        )

        self.pool = CompactBilinearPooling(hidden, hidden, config.pooling_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.pooling_dim, config.num_answers),
        )

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                nn.init.zeros_(param) if name.startswith("bias") else nn.init.kaiming_normal_(param)

    def forward(
        self,
        question_input_ids: torch.LongTensor,
        image_features: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        answer_scores: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            question_input_ids (`torch.LongTensor` of shape `(batch, max_question_length)`):
                Question word ids.
            image_features (`torch.FloatTensor` of shape `(batch, num_regions, image_feature_dim)`):
                Pre-extracted region features.
            labels (`torch.LongTensor` of shape `(batch,)`, *optional*):
                Index of the correct answer in the answer vocabulary.
            answer_scores (`torch.FloatTensor` of shape `(batch, num_answers)`, *optional*):
                Soft targets: the VQA accuracy each answer would earn, given the
                ten human annotations. Used instead of `labels` when
                `config.soft_targets` is set, since the metric itself is graded.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        question = self.question_encoder(question_input_ids)
        image = self.image_encoder(image_features)

        # Keep attention off the padded question slots; see [`HighOrderAttentionForVQA`].
        masks = [question_input_ids != 0, None] if self.config.mask_padding else None
        attended = self.attention(question, image, masks=masks, return_weights=True)
        attended, weights = attended if output_attentions else (attended[0], None)
        pooled_question, pooled_image = attended

        fused = F.normalize(signed_sqrt(self.pool(pooled_question, pooled_image)), dim=-1)
        logits = self.classifier(fused)

        loss = None
        if self.config.loss_type == "ce":
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
        elif answer_scores is not None:
            if self.config.loss_type == "bce":
                # Multi-label regression: each answer is scored on its own, so
                # several can be right at once and a near-miss keeps partial
                # credit. Scaled back up by the vocabulary size that
                # `binary_cross_entropy_with_logits` averaged over, so the signal
                # does not shrink as the vocabulary grows.
                loss = F.binary_cross_entropy_with_logits(logits, answer_scores) * logits.size(-1)
            else:
                # The same targets renormalized to a distribution, so answers compete.
                mass = answer_scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                loss = -((answer_scores / mass) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        elif self.training:
            raise ValueError(
                f"loss_type={self.config.loss_type!r} needs answer_scores, and none were given. Build the "
                "dataset with num_answers set so it supplies them."
            )
        elif labels is not None:
            # Evaluating a graded model on data that carries only a hard label.
            loss = F.cross_entropy(logits, labels)

        pooled = dict(zip(OPEN_ENDED_MODALITIES, attended))
        if not return_dict:
            output = (logits, pooled) + ((tuple(weights),) if weights is not None else ())
            return ((loss,) + output) if loss is not None else output

        return OpenEndedVQAOutput(
            loss=loss,
            logits=logits,
            pooled_modalities=pooled,
            attentions=tuple(weights) if weights is not None else None,
        )
