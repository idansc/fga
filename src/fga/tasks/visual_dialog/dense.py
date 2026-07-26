"""Finetuning against dense relevance annotations, to optimize NDCG.

The sparse objective treats exactly one of the 100 candidates as correct and the
other 99 as equally wrong. That is what MRR measures, but it is a poor match for
how the answers actually look: for "is it daytime?", *yes*, *yeah*, *it is* and
*yes it is* are all right, and the sparse label arbitrarily picks one. NDCG scores
against the graded relevance five annotators assigned to every candidate, so a
model trained on the sparse label is optimising the wrong thing for it.

This module trains on that graded signal instead, over the 2,000-image subset of
*train* for which dense annotations were released. The val annotations are never
trained on -- they are the evaluation set.

The objective is a soft-label cross entropy: the relevance vector is normalized
into a distribution and matched against the model's log-softmax over candidates,

    loss = -sum_i p_i * log q_i,    p = relevance / sum(relevance)

which is the cross entropy the sparse loss already computes, generalized from a
one-hot target to a graded one. Optionally it is mixed with the sparse loss so
the ranking of the single ground-truth answer is not discarded entirely; that
trade-off is exactly the tension reported in
https://github.com/idansc/mrr-ndcg.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer

from .data import DenseAnnotationsReader, VisDialDataset

__all__ = ["DenseVisDialDataset", "DenseFinetuneTrainer", "dense_soft_cross_entropy"]


def dense_soft_cross_entropy(logits: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """Cross entropy against a graded relevance distribution.

    Args:
        logits: `(batch, num_options)` model scores.
        relevance: `(batch, num_options)` non-negative human relevance.

    Rows whose relevance is entirely zero carry no signal and are skipped rather
    than producing a uniform target, which would actively flatten the ranking.
    """
    mass = relevance.sum(dim=-1, keepdim=True)
    keep = (mass > 0).squeeze(-1)
    if not keep.any():
        return logits.sum() * 0.0

    target = relevance[keep] / mass[keep]
    log_probs = F.log_softmax(logits[keep], dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


class DenseVisDialDataset(Dataset):
    """The dialog rounds that carry dense relevance annotations.

    One example per annotated image: the round the annotators judged, paired with
    its 100-way relevance vector. Everything else matches [`VisDialDataset`], so
    the same collator and model apply.

    Args:
        dataset: the underlying split, which must be the split the annotations
            describe.
        dense_annotations_path: the dense annotations JSON for that split.
        image_ids: image id per image, in dataset order, from
            `visdial_params.json`.
        keep_sparse_labels: also return `labels`, the sparse ground-truth index,
            so a mixed objective can use both.
    """

    def __init__(
        self,
        dataset: VisDialDataset,
        dense_annotations_path: str,
        image_ids: List[int],
        keep_sparse_labels: bool = True,
    ):
        self.dataset = dataset
        self.keep_sparse_labels = keep_sparse_labels
        reader = DenseAnnotationsReader(dense_annotations_path)

        num_images = len(dataset.cap)
        rounds = dataset.n_qa_per_dial

        self.examples: List[Dict[str, Any]] = []
        for image_index, image_id in enumerate(image_ids[:num_images]):
            if image_id not in reader:
                continue
            entry = reader[image_id]
            # round_id is 1-based; the flat dataset index is image * rounds + round.
            round_index = int(entry["round_id"]) - 1
            if not 0 <= round_index < rounds:
                continue
            self.examples.append(
                {
                    "index": image_index * rounds + round_index,
                    "relevance": np.asarray(reader.relevance_of(entry), dtype=np.float32),
                }
            )

        if not self.examples:
            raise ValueError(
                f"No dense annotations matched this split. Checked {len(image_ids[:num_images])} images "
                f"against {len(reader)} annotated ones -- are the image ids and the split the same?"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        example = self.examples[i]
        item = dict(self.dataset[example["index"]])
        if not self.keep_sparse_labels:
            item.pop("labels", None)
        item["relevance"] = example["relevance"]
        return item


class DenseFinetuneTrainer(Trainer):
    """[`Trainer`] whose loss is the graded relevance, optionally mixed with the sparse one.

    Args:
        dense_weight: weight on the dense soft-label term.
        sparse_weight: weight on the ordinary one-hot cross entropy. Set to 0 for
            a purely NDCG-driven objective; a small value retains some of the MRR
            behaviour, since training only on graded relevance is known to trade
            MRR away.
    """

    def __init__(self, *args, dense_weight: float = 1.0, sparse_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        relevance = inputs.pop("relevance", None)
        labels = inputs.get("labels")
        if self.sparse_weight == 0:
            inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = logits.sum() * 0.0
        if relevance is not None and self.dense_weight:
            loss = loss + self.dense_weight * dense_soft_cross_entropy(logits, relevance.to(logits.dtype))
        if self.sparse_weight and labels is not None:
            loss = loss + self.sparse_weight * F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def load_relevance_matrix(path: str, image_ids: List[int], num_options: int = 100) -> Optional[np.ndarray]:
    """Relevance for each image in dataset order, zero where unannotated."""
    with open(path) as handle:
        entries = json.load(handle)
    by_id = {entry["image_id"]: entry for entry in entries}

    matrix = np.zeros((len(image_ids), num_options), dtype=np.float32)
    for i, image_id in enumerate(image_ids):
        entry = by_id.get(image_id)
        if entry is not None:
            matrix[i] = np.asarray(DenseAnnotationsReader.relevance_of(entry), dtype=np.float32)
    return matrix
