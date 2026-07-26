"""Visual Question Answering with high-order attention.

Two settings are provided:

* [`HighOrderAttentionForVQA`] — **multiple choice**. Three modalities: question
  words, image regions and the candidate answers. Because there are three, it can
  use a **ternary** factor scoring (region, word, answer) triples directly; a
  triple can be jointly consistent while no two of its parts stand out on their
  own, which pairwise factors cannot express.
* [`OpenEndedVQAModel`] — **open ended**. No candidates are given, so the model
  attends question and image only and classifies over the answer vocabulary.
  With two modalities there is no ternary factor to apply, and the answer can no
  longer steer where the model looks.

Ported from https://github.com/idansc/HighOrderAtten, "High-Order Attention Models
for Visual Question Answering" (NeurIPS 2017).
"""

from .modeling_hoa import (
    MODALITY_NAMES,
    HighOrderAttentionConfig,
    HighOrderAttentionForVQA,
    HighOrderAttentionOutput,
    QuestionEncoder,
)
from .modeling_open_ended import (
    OPEN_ENDED_MODALITIES,
    OpenEndedVQAConfig,
    OpenEndedVQAModel,
    OpenEndedVQAOutput,
)
from .pooling import CompactBilinearPooling, count_sketch, signed_sqrt

__all__ = [
    "MODALITY_NAMES",
    "OPEN_ENDED_MODALITIES",
    "CompactBilinearPooling",
    "HighOrderAttentionConfig",
    "HighOrderAttentionForVQA",
    "HighOrderAttentionOutput",
    "OpenEndedVQAConfig",
    "OpenEndedVQAModel",
    "OpenEndedVQAOutput",
    "QuestionEncoder",
    "count_sketch",
    "signed_sqrt",
]
