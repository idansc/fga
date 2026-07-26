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

__all__ = [
    "DenseVisDialDataset",
    "DenseFinetuneTrainer",
    "approx_ndcg_loss",
    "dense_soft_cross_entropy",
]


def approx_ndcg_loss(
    logits: torch.Tensor,
    relevance: torch.Tensor,
    temperature: float = 1.0,
    normalize_scores: bool = True,
) -> torch.Tensor:
    """A differentiable approximation of NDCG, following Qin et al. (2010).

    Soft cross entropy matches a distribution, which is not quite the objective:
    it spends as much effort separating ranks 80 and 81 as ranks 1 and 2, while
    NDCG discounts by position and mostly cares about the top. ApproxNDCG instead
    smooths the *rank* itself, replacing the hard sort with

        rank(i) ~= 1 + sum_{j != i} sigmoid((s_j - s_i) / temperature)

    which is differentiable, and plugs it straight into the NDCG definition. As
    the temperature falls this converges on the true metric, at the cost of
    vanishing gradients.

    Gains are linear in the relevance, matching the official VisDial NDCG
    implementation, rather than the `2^rel - 1` used elsewhere in the ranking
    literature -- the loss has to approximate the metric actually being reported.

    Args:
        logits: `(batch, num_options)` model scores.
        relevance: `(batch, num_options)` non-negative human relevance.
        temperature: smoothing of the rank approximation; lower is sharper.
        normalize_scores: divide each row's scores by their standard deviation
            first, so that `temperature` means the same thing regardless of how
            spread the model's logits happen to be. Without it, a confident model
            saturates every sigmoid and the rank approximation stops responding.
            It does not change the ranking being scored.

    Note that this loss has a much smaller natural gradient magnitude than the
    soft cross entropy -- NDCG gains are bounded by 1 and the position discount is
    flat -- so it needs a scale-invariant optimizer. Adam is fine; plain SGD at the
    same learning rate will appear not to train.

    Returns: `1 - approximate NDCG`, so that lower is better.
    """
    mass = relevance.sum(dim=-1, keepdim=True)
    keep = (mass > 0).squeeze(-1)
    if not keep.any():
        return logits.sum() * 0.0

    scores = logits[keep]
    gains = relevance[keep]

    if normalize_scores:
        scores = scores / scores.std(dim=-1, keepdim=True).clamp(min=1e-6)

    # Smooth rank of candidate i: how many candidates are estimated to outrank it.
    # pairwise[b, i, j] = (s_j - s_i) / temperature, so the sum must run over j.
    pairwise = (scores.unsqueeze(1) - scores.unsqueeze(2)) / temperature
    approx_rank = 1.0 + (torch.sigmoid(pairwise).sum(dim=2) - 0.5)

    discounted = gains / torch.log2(approx_rank + 1.0)
    dcg = discounted.sum(dim=-1)

    # Ideal DCG uses the true sort, and needs no gradient.
    ideal_gains, _ = gains.sort(dim=-1, descending=True)
    positions = torch.arange(gains.size(-1), device=gains.device, dtype=gains.dtype)
    ideal_dcg = (ideal_gains / torch.log2(positions + 2.0)).sum(dim=-1)

    return (1.0 - dcg / ideal_dcg.clamp(min=1e-12)).mean()


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
        loss_kind: `"soft_ce"` for the soft-label cross entropy (what the VisDial
            literature generally uses), or `"approx_ndcg"` to optimize a smooth
            approximation of the metric itself.
        approx_ndcg_temperature: rank smoothing, when `loss_kind="approx_ndcg"`.
        dense_weight: weight on the dense term.
        sparse_weight: weight on the ordinary one-hot cross entropy. Set to 0 for
            a purely NDCG-driven objective; a small value retains some of the MRR
            behaviour, since training only on graded relevance is known to trade
            MRR away.
    """

    def __init__(
        self,
        *args,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.0,
        loss_kind: str = "soft_ce",
        approx_ndcg_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_kind not in ("soft_ce", "approx_ndcg"):
            raise ValueError(f"loss_kind must be 'soft_ce' or 'approx_ndcg', got {loss_kind!r}")
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.loss_kind = loss_kind
        self.approx_ndcg_temperature = approx_ndcg_temperature

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        relevance = inputs.pop("relevance", None)
        if relevance is None and self.dense_weight and model.training:
            # Trainer strips columns absent from the model signature when
            # `remove_unused_columns` is left on, which silently deletes the
            # relevance and leaves a constant-zero dense term. Only a problem
            # during training: the evaluation set carries no relevance by design,
            # and `prediction_step` routes through here too.
            raise ValueError(
                "dense_weight > 0 but no 'relevance' in the training batch. Set "
                "TrainingArguments(remove_unused_columns=False) so the collator's "
                "extra column survives."
            )
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = logits.sum() * 0.0
        if relevance is not None and self.dense_weight:
            relevance = relevance.to(logits.dtype)
            if self.loss_kind == "approx_ndcg":
                dense_term = approx_ndcg_loss(logits, relevance, self.approx_ndcg_temperature)
            else:
                dense_term = dense_soft_cross_entropy(logits, relevance)
            loss = loss + self.dense_weight * dense_term
        if labels is not None and (self.sparse_weight or relevance is None):
            # During evaluation there is no relevance, so the reported loss is the
            # ordinary cross entropy and stays comparable across the sweep.
            weight = self.sparse_weight if relevance is not None else 1.0
            loss = loss + weight * F.cross_entropy(logits, labels)

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
