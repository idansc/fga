"""Data pipeline for multiple-choice VQA v1.

`scripts/prepare_vqa.py` turns the official VQA v1 release into three files:

    vqa_mc.h5        tokenized questions, choice ids, labels, per split
    features.h5      image features, one row per COCO id, id-indexed
    vocab.json       question vocabulary and the answer vocabulary

[`VQAMultipleChoiceDataset`] joins them at access time, yielding dictionaries
keyed like [`HighOrderAttentionForVQA.forward`].

Answer ids: 0 is reserved for padding and out-of-vocabulary choices (the choice
embedding uses `padding_idx=0`), real answers are `1..num_answers`. Training
rows whose correct answer falls outside the vocabulary are dropped by the prep
script; evaluation keeps every row, since prediction is an argmax restricted to
the 18 given choices.
"""

import json
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["VQAMultipleChoiceDataset", "VQACollator", "vqa_accuracy"]


def vqa_accuracy(prediction: str, human_answers) -> float:
    """The official VQA metric: full credit once three of ten humans agree."""
    return min(sum(a == prediction for a in human_answers) / 3.0, 1.0)


class VQAMultipleChoiceDataset(Dataset):
    """One example per question: token ids, 18 candidate answers, image features.

    Args:
        vqa_h5_path: output of `scripts/prepare_vqa.py`.
        features_h5_path: id-indexed image features.
        split: `"train"` or `"val"`.
        in_memory: hold the feature table in RAM (~18 GB as float16).
    """

    def __init__(self, vqa_h5_path: str, features_h5_path: str, split: str, in_memory: bool = False):
        self.split = split
        self.features_h5_path = features_h5_path
        self._features: Optional[h5py.File] = None

        with h5py.File(vqa_h5_path, "r") as h5:
            self.questions = h5[f"{split}_questions"][:]
            self.choices = h5[f"{split}_choices"][:]
            self.labels = h5[f"{split}_labels"][:]
            self.feature_rows = h5[f"{split}_feature_rows"][:]
            self.question_ids = h5[f"{split}_question_ids"][:]

        self.features_in_memory = None
        if in_memory:
            with h5py.File(features_h5_path, "r") as h5:
                self.features_in_memory = h5["features"][:]

    def __len__(self) -> int:
        return len(self.questions)

    def _feature(self, row: int) -> np.ndarray:
        if self.features_in_memory is not None:
            return self.features_in_memory[row]
        if self._features is None:  # reopened lazily per worker; handles do not fork
            self._features = h5py.File(self.features_h5_path, "r", swmr=True)
        return self._features["features"][row]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "question_input_ids": self.questions[index].astype(np.int64),
            "choice_input_ids": self.choices[index].astype(np.int64),
            "image_features": np.asarray(self._feature(self.feature_rows[index]), dtype=np.float32),
            "labels": np.int64(self.labels[index]),
        }


class VQACollator:
    """Stack examples into batched tensors."""

    def __call__(self, features):
        batch = {}
        for key in features[0]:
            batch[key] = torch.from_numpy(np.stack([np.asarray(f[key]) for f in features]))
        return batch


def load_vocab(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)
