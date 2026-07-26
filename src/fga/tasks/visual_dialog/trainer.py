"""Training and evaluation helpers built on [`transformers.Trainer`].

The custom pieces are all evaluation-side: Visual Dialog is scored per *image*
(NDCG uses one densely-annotated round per image) while the dataset is iterated
per *round*, so predictions have to be folded back before NDCG can be computed,
and the EvalAI submission file needs the per-image round counts.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from .data import DenseAnnotationsReader
from .metrics import ranks_to_submission

__all__ = ["FGATrainingArguments", "FGATrainer", "load_dense_annotations"]


@dataclass
class FGATrainingArguments(TrainingArguments):
    """[`TrainingArguments`] plus the two knobs specific to VisDial evaluation."""

    num_rounds: int = field(
        default=10,
        metadata={"help": "Dialog rounds per image; used to fold round-level predictions back per image."},
    )
    write_submission: bool = field(
        default=False,
        metadata={"help": "Write an EvalAI-format submission JSON after every evaluation/prediction."},
    )


def load_dense_annotations(path: str, image_ids: Sequence[int]) -> Dict[str, np.ndarray]:
    """Align dense val annotations to the dataset's image order.

    Args:
        path: `visdial_1.0_val_dense_annotations.json`.
        image_ids: image ids in dataset order, from `visdial_params.json`.

    Returns: `{"relevance": (num_images, 100), "round_ids": (num_images,)}`.
        Images without an annotation get zero relevance and are effectively
        skipped by NDCG.
    """
    reader = DenseAnnotationsReader(path)
    num_options = len(reader.relevance_of(reader[image_ids[0]])) if image_ids else 100

    relevance = np.zeros((len(image_ids), num_options), dtype=np.float32)
    round_ids = np.ones(len(image_ids), dtype=np.int64)
    missing = 0
    for i, image_id in enumerate(image_ids):
        if image_id not in reader:
            missing += 1
            continue
        entry = reader[image_id]
        relevance[i] = np.asarray(reader.relevance_of(entry), dtype=np.float32)
        round_ids[i] = int(entry["round_id"])
    if missing:
        print(f"Warning: {missing}/{len(image_ids)} images have no dense annotation; they contribute 0 to NDCG.")
    return {"relevance": relevance, "round_ids": round_ids}


class FGATrainer(Trainer):
    """[`Trainer`] that can also emit EvalAI submission files.

    Args:
        image_ids: image id per image of the evaluated split, in dataset order.
        num_rounds_per_image: number of real rounds per dialog. The test split is
            cut at a random round and only that round is scored.

    The remaining arguments are forwarded to [`Trainer`].
    """

    def __init__(
        self,
        *args,
        image_ids: Optional[Sequence[int]] = None,
        num_rounds_per_image: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_ids = list(image_ids) if image_ids is not None else None
        self.num_rounds_per_image = list(num_rounds_per_image) if num_rounds_per_image is not None else None

    def write_submission(self, scores: np.ndarray, filename: str = "submission.json") -> Optional[str]:
        """Write ranked predictions in the format the challenge server expects."""
        if self.image_ids is None:
            print("Skipping submission: no image_ids were supplied to FGATrainer.")
            return None

        num_rounds = getattr(self.args, "num_rounds", 10)
        num_images = len(scores) // num_rounds
        image_ids = self.image_ids[:num_images]
        rounds_per_image = (
            self.num_rounds_per_image[:num_images]
            if self.num_rounds_per_image is not None
            else [num_rounds] * num_images
        )

        submission = ranks_to_submission(
            torch.as_tensor(np.asarray(scores[: num_images * num_rounds], dtype=np.float32)),
            image_ids=image_ids,
            num_rounds_per_image=rounds_per_image,
            num_rounds=num_rounds,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, filename)
        with open(path, "w") as handle:
            json.dump(submission, handle)
        print(f"Wrote {len(submission)} ranked rounds to {path}")
        return path
