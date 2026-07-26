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
    """Word embeddings, a trigram convolution, and a bidirectional LSTM.

    The trigram convolution is what the original uses to give each position local
    phrase context before the recurrence.
    """

    def __init__(self, config: HighOrderAttentionConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.word_embed_dim, padding_idx=0)
        self.trigram = nn.Conv1d(config.word_embed_dim, config.word_embed_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            config.word_embed_dim,
            config.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embedded = self.dropout(torch.tanh(self.embedding(input_ids)))
        local = torch.tanh(self.trigram(embedded.transpose(1, 2))).transpose(1, 2)
        sequence, _ = self.lstm(local)
        return self.dropout(sequence)


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
            ternary_interactions=[(0, 1, 2)] if config.use_ternary else None,
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
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        question = self.question_encoder(question_input_ids)
        image = self.image_encoder(image_features)
        choices = self.choice_dropout(torch.tanh(self.choice_embedding(choice_input_ids)))

        attended = self.attention(question, image, choices, return_weights=True)
        attended, weights = attended if output_attentions else (attended[0], None)
        pooled_question, pooled_image, pooled_answer = attended

        answer_image = self._fuse(self.pool_answer_image(pooled_answer, pooled_image))
        question_image = self._fuse(self.pool_question_image(pooled_question, pooled_image))
        joint = self._fuse(self.pool_joint(answer_image, question_image))

        logits = self.classifier(joint)
        loss = F.cross_entropy(logits, labels) if labels is not None else None

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
