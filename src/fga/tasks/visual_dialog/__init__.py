"""Visual Dialog: the reference application of factor-graph attention.

Six utilities — the 100 candidate answers, the question, the caption, the image
regions, and the question and answer of each history round — are attended jointly,
and every candidate answer is scored against the fused context.

This is what the CVPR'19 paper reports; see [`fga.attention`] for the general
module underneath, which carries none of this task's assumptions.
"""

from .configuration_fga import FGAConfig
from .data import (
    DenseAnnotationsReader,
    VisDialCollator,
    VisDialDataset,
    build_hf_dataset,
    image_ids_from_params,
    load_visdial_params,
    vocab_size_from_params,
)
from .encoders import (
    FGATextEncoder,
    LSTMTextEncoder,
    build_text_encoder,
    register_text_encoder,
)
from .metrics import build_compute_metrics, ndcg, ranks_to_submission, scores_to_ranks, sparse_metrics
from .modeling_fga import (
    UTILITY_NAMES,
    FGAForVisualDialog,
    FGAForVisualDialogOutput,
    FGAModel,
    FGAModelOutput,
)
from .trainer import FGATrainer, FGATrainingArguments, load_dense_annotations

__all__ = [
    "UTILITY_NAMES",
    "DenseAnnotationsReader",
    "FGAConfig",
    "FGAForVisualDialog",
    "FGAForVisualDialogOutput",
    "FGAModel",
    "FGAModelOutput",
    "FGATextEncoder",
    "FGATrainer",
    "FGATrainingArguments",
    "LSTMTextEncoder",
    "VisDialCollator",
    "VisDialDataset",
    "build_compute_metrics",
    "build_hf_dataset",
    "build_text_encoder",
    "image_ids_from_params",
    "load_dense_annotations",
    "load_visdial_params",
    "ndcg",
    "ranks_to_submission",
    "register_text_encoder",
    "scores_to_ranks",
    "sparse_metrics",
    "vocab_size_from_params",
]
