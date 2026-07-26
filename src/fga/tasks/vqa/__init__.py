"""Visual Question Answering with high-order attention.

A PyTorch port of https://github.com/idansc/HighOrderAtten — "High-Order Attention
Models for Visual Question Answering" (NeurIPS 2017) — rebuilt on
[`fga.attention`].

Three modalities are attended jointly: question words, image regions and
multiple-choice answers. The distinguishing piece is the **ternary** factor, which
scores (region, word, answer) triples directly. A triple can be jointly consistent
while no two of its parts stand out on their own, so pairwise factors cannot
express it.
"""

from .modeling_hoa import (
    MODALITY_NAMES,
    HighOrderAttentionConfig,
    HighOrderAttentionForVQA,
    HighOrderAttentionOutput,
    QuestionEncoder,
)
from .pooling import CompactBilinearPooling, count_sketch, signed_sqrt

__all__ = [
    "MODALITY_NAMES",
    "CompactBilinearPooling",
    "HighOrderAttentionConfig",
    "HighOrderAttentionForVQA",
    "HighOrderAttentionOutput",
    "QuestionEncoder",
    "count_sketch",
    "signed_sqrt",
]
