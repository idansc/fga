"""Text-to-video retrieval.

A query and a video are each a sequence — words and clips — and attention decides
which clips and which words matter for each other before the two are compared.
Training is contrastive rather than classification: a true pair must outscore the
hardest mismatched pair in the batch by a margin.

Ported from https://github.com/AmeenAli/VideoMatch.
"""

from .modeling_videomatch import (
    MODALITY_NAMES,
    VideoMatchConfig,
    VideoMatchModel,
    VideoMatchOutput,
    contrastive_loss,
)

__all__ = [
    "MODALITY_NAMES",
    "VideoMatchConfig",
    "VideoMatchModel",
    "VideoMatchOutput",
    "contrastive_loss",
]
