"""Multiple-choice Visual Question Answering with high-order attention.

Three modalities are attended jointly: the question words, the image regions, and
the multiple-choice answers. What distinguishes it from the Visual Dialog model is
the **ternary** factor — a potential over (region, word, answer) triples, which no
pair of modalities can express on its own.

The attention itself is [`fga.attention.FactorGraphAttention`] with a ternary
interaction declared, so this task adds only its encoders and its scoring head.
The potentials therefore follow FGA's conventions — normalized embeddings, a
batch-normalized interaction grid, convolutional marginalization — rather than the
original's `tanh` and elementwise scaling.

----

Ported from https://github.com/idansc/HighOrderAtten, "High-Order Attention Models
for Visual Question Answering" (Schwartz, Schwing and Hazan, NeurIPS 2017),
originally written in Lua/Torch.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ...attention import FactorGraphAttention
from .pooling import CompactBilinearPooling, signed_sqrt

__all__ = ["HighOrderAttentionConfig", "HighOrderAttentionForVQA", "HighOrderAttentionOutput"]

#: Modality order, fixing the meaning of the indices below.
MODALITY_NAMES = ("question", "image", "answer")


class HighOrderAttentionConfig(PretrainedConfig):
    r"""Configuration for [`HighOrderAttentionForVQA`].

    Args:
        vocab_size (`int`, *optional*, defaults to 12604):
            Question vocabulary, including the padding id 0.
        num_answers (`int`, *optional*, defaults to 3000):
            Size of the answer vocabulary the model classifies over, and of the
            embedding table the multiple-choice candidates are looked up in.
        hidden_size (`int`, *optional*, defaults to 512):
            Shared dimension the three modalities are projected to.
        word_embed_dim (`int`, *optional*, defaults to 512):
            Question word embedding dimension.
        image_feature_dim (`int`, *optional*, defaults to 2048):
            Dimension of the pre-extracted region features.
        num_regions (`int`, *optional*, defaults to 196):
            Image regions, 14x14 for the conv grid used in the paper.
        max_question_length (`int`, *optional*, defaults to 15):
            Question words after truncation.
        num_choices (`int`, *optional*, defaults to 18):
            Multiple-choice candidates per question.
        pooling_dim (`int`, *optional*, defaults to 16000):
            Compact Bilinear Pooling sketch size.
        use_ternary (`bool`, *optional*, defaults to `True`):
            Include the three-way factor. Setting this to `False` reduces the
            model to pairwise attention, which is the ablation the paper reports.
        dropout (`float`, *optional*, defaults to 0.5):
            Dropout on the encoders.
        classifier_dropout (`float`, *optional*, defaults to 0.3):
            Dropout before the answer classifier.
        loss_type (`str`, *optional*, defaults to `"ce"`):
            `"ce"` supervises one label, which is the objective the paper trains
            and so the default here. `"soft_ce"` and `"bce"` instead supervise the
            VQA score every answer earns from the ten annotators, and need
            `answer_scores`; see [`OpenEndedVQAConfig`]. On open-ended VQA the
            graded targets are worth about a point, so they are worth trying here
            too — but changing the default would quietly stop this being a
            reproduction.
        mask_padding (`bool`, *optional*, defaults to `True`):
            Keep attention off padded question words and candidate slots, and
            zero the encoder state at padded positions. Set to `False` for the
            ablation — it costs accuracy, since the padded positions then absorb
            most of the attention mass.
    """

    model_type = "high_order_attention"

    def __init__(
        self,
        vocab_size: int = 12604,
        num_answers: int = 3000,
        hidden_size: int = 512,
        word_embed_dim: int = 512,
        image_feature_dim: int = 2048,
        num_regions: int = 196,
        max_question_length: int = 15,
        num_choices: int = 18,
        pooling_dim: int = 16000,
        use_ternary: bool = True,
        dropout: float = 0.5,
        classifier_dropout: float = 0.3,
        loss_type: str = "ce",
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
        self.num_choices = num_choices
        self.pooling_dim = pooling_dim
        self.use_ternary = use_ternary
        self.dropout = dropout
        self.classifier_dropout = classifier_dropout
        if loss_type not in ("bce", "soft_ce", "ce"):
            raise ValueError(f"loss_type must be one of bce, soft_ce, ce; got {loss_type!r}.")
        self.loss_type = loss_type
        self.mask_padding = mask_padding
        kwargs.setdefault("pad_token_id", 0)
        super().__init__(**kwargs)


@dataclass
class HighOrderAttentionOutput(ModelOutput):
    """Output of [`HighOrderAttentionForVQA`].

    Args:
        loss (`torch.FloatTensor`, *optional*): cross entropy, when `labels` is given.
        logits (`torch.FloatTensor` of shape `(batch, num_answers)`): answer scores.
        pooled_modalities (`Dict[str, torch.FloatTensor]`, *optional*):
            The attended question, image and answer representations.
        attentions (`Tuple[torch.FloatTensor]`, *optional*):
            Attention over each modality, in [`MODALITY_NAMES`] order — the
            question-word, image-region and candidate-answer distributions the
            paper visualizes.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    pooled_modalities: Optional[dict] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class QuestionEncoder(nn.Module):
    """Word embeddings and a trigram convolution, each read by its own LSTM.

    Two unidirectional recurrences run in parallel -- one over the word
    embeddings, one over the trigram convolution -- and are concatenated, so the
    encoder carries word-level and phrase-level state side by side. That is the
    original's `rnn1(embed)` / `rnn2(trigram)` pair; an earlier version of this
    port ran a single bidirectional LSTM over the convolution alone, which drops
    the word-level stream entirely.

    Both streams are zeroed at padded positions, matching the original's
    `maskzero`. Questions are padded on the right, so a padded step never
    precedes a real one and zeroing the output is equivalent to suppressing the
    recurrence there.
    """

    def __init__(self, config: HighOrderAttentionConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.word_embed_dim, padding_idx=0)
        self.trigram = nn.Conv1d(config.word_embed_dim, config.word_embed_dim, kernel_size=3, padding=1)
        self.word_lstm = nn.LSTM(config.word_embed_dim, config.hidden_size // 2, batch_first=True)
        self.phrase_lstm = nn.LSTM(config.word_embed_dim, config.hidden_size // 2, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.mask_padding = config.mask_padding

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embedded = self.dropout(torch.tanh(self.embedding(input_ids)))
        local = torch.tanh(self.trigram(embedded.transpose(1, 2))).transpose(1, 2)

        word, _ = self.word_lstm(embedded)
        phrase, _ = self.phrase_lstm(local)
        sequence = self.dropout(torch.cat((word, phrase), dim=-1))
        if not self.mask_padding:
            return sequence
        return sequence * (input_ids != 0).unsqueeze(-1)


class HighOrderAttentionForVQA(PreTrainedModel):
    """Multiple-choice VQA by attending jointly over question, image and answers.

    Example:

    ```python
    >>> from fga.tasks.vqa import HighOrderAttentionConfig, HighOrderAttentionForVQA
    >>> model = HighOrderAttentionForVQA(HighOrderAttentionConfig())
    >>> outputs = model(
    ...     question_input_ids=question,     # (batch, max_question_length)
    ...     image_features=regions,          # (batch, num_regions, image_feature_dim)
    ...     choice_input_ids=choices,        # (batch, num_choices)
    ... )  # doctest: +SKIP
    ```
    """

    config_class = HighOrderAttentionConfig
    base_model_prefix = "hoa"
    main_input_name = "question_input_ids"

    def __init__(self, config: HighOrderAttentionConfig):
        super().__init__(config)
        hidden = config.hidden_size

        self.question_encoder = QuestionEncoder(config)
        self.image_encoder = nn.Sequential(
            nn.Linear(config.image_feature_dim, hidden),
            nn.Tanh(),
            nn.Dropout(config.dropout),
        )
        self.choice_embedding = nn.Embedding(config.num_answers, hidden, padding_idx=0)
        self.choice_dropout = nn.Dropout(config.dropout)

        self.attention = FactorGraphAttention(
            embed_dims=[hidden, hidden, hidden],
            num_entities=[config.max_question_length, config.num_regions, config.num_choices],
            modality_names=list(MODALITY_NAMES),
            ternary_interactions=[MODALITY_NAMES] if config.use_ternary else None,
        )

        # The original fuses answer-image and question-image, then fuses those.
        self.pool_answer_image = CompactBilinearPooling(hidden, hidden, config.pooling_dim)
        self.pool_question_image = CompactBilinearPooling(hidden, hidden, config.pooling_dim)
        self.pool_joint = CompactBilinearPooling(config.pooling_dim, config.pooling_dim, config.pooling_dim)

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
                if name.startswith("bias"):
                    nn.init.zeros_(param)
                else:
                    nn.init.kaiming_normal_(param)

    @staticmethod
    def _fuse(pooled: torch.Tensor) -> torch.Tensor:
        """Signed square root then L2, as the VQA pooling literature prescribes."""
        return F.normalize(signed_sqrt(pooled), dim=-1)

    def forward(
        self,
        question_input_ids: torch.LongTensor,
        image_features: torch.FloatTensor,
        choice_input_ids: torch.LongTensor,
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
            choice_input_ids (`torch.LongTensor` of shape `(batch, num_choices)`):
                Answer-vocabulary ids of the multiple-choice candidates.
            labels (`torch.LongTensor` of shape `(batch,)`, *optional*):
                Index into the answer vocabulary of the correct answer.
            answer_scores (`torch.FloatTensor` of shape `(batch, num_answers)`, *optional*):
                The VQA score each answer earns from this question's annotators.
                Required by the graded objectives.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        question = self.question_encoder(question_input_ids)
        image = self.image_encoder(image_features)
        choices = self.choice_dropout(torch.tanh(self.choice_embedding(choice_input_ids)))

        # Questions are padded to 15 words and the candidate list to 18 slots.
        # Padding is not inert -- an encoder still emits a state there, and the
        # unmasked model put 68% of its question attention and 32% of its answer
        # attention on those slots -- so the empty entities are excluded from the
        # softmax. The original masks the question only; the candidates are
        # masked here too, since attending to an absent answer cannot help.
        # Regions have no padding.
        masks = None
        if self.config.mask_padding:
            masks = [question_input_ids != 0, None, choice_input_ids != 0]
        attended = self.attention(question, image, choices, masks=masks, return_weights=True)
        attended, weights = attended if output_attentions else (attended[0], None)
        pooled_question, pooled_image, pooled_answer = attended

        answer_image = self._fuse(self.pool_answer_image(pooled_answer, pooled_image))
        question_image = self._fuse(self.pool_question_image(pooled_question, pooled_image))
        joint = self._fuse(self.pool_joint(answer_image, question_image))

        logits = self.classifier(joint)

        loss = None
        if self.config.loss_type == "ce":
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
        elif answer_scores is not None:
            if self.config.loss_type == "bce":
                loss = F.binary_cross_entropy_with_logits(logits, answer_scores) * logits.size(-1)
            else:
                mass = answer_scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                loss = -((answer_scores / mass) * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        elif self.training:
            raise ValueError(
                f"loss_type={self.config.loss_type!r} needs answer_scores, and none were given. Build the "
                "dataset with num_answers set so it supplies them."
            )
        elif labels is not None:
            loss = F.cross_entropy(logits, labels)

        pooled = dict(zip(MODALITY_NAMES, (pooled_question, pooled_image, pooled_answer)))
        if not return_dict:
            output = (logits, pooled) + ((tuple(weights),) if weights is not None else ())
            return ((loss,) + output) if loss is not None else output

        return HighOrderAttentionOutput(
            loss=loss,
            logits=logits,
            pooled_modalities=pooled,
            attentions=tuple(weights) if weights is not None else None,
        )
