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
import re
from typing import Any, Dict, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["VQAMultipleChoiceDataset", "VQACollator", "normalize_answer", "vqa_accuracy"]

#: The official evaluation's rewrites, so that "two" and "2" are the same answer.
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "isnt": "isn't", "itd": "it'd", "itll": "it'll", "lets": "let's",
    "maam": "ma'am", "mightve": "might've", "mustve": "must've", "shant": "shan't",
    "shed": "she'd", "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "thats": "that's", "theres": "there's", "theyd": "they'd", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't", "werent": "weren't",
    "whatre": "what're", "whats": "what's", "wheres": "where's", "whos": "who's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")
_THOUSANDS = re.compile(r"(\d),(\d)")
_PUNCTUATION = ";/[]\"{}()=+\\_-><@`,?!"


def normalize_answer(answer: str) -> str:
    """Lower-case, strip punctuation and articles, spell numbers as digits."""
    answer = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    answer = _THOUSANDS.sub(r"\1\2", answer)
    answer = _PERIOD.sub("", answer)
    answer = answer.translate(str.maketrans("", "", _PUNCTUATION))
    words = [_CONTRACTIONS.get(w, _NUMBERS.get(w, w)) for w in answer.split()]
    return " ".join(w for w in words if w not in _ARTICLES)


def vqa_accuracy(prediction: str, human_answers: Sequence[str], normalize: bool = True) -> float:
    """The official VQA metric.

    Full credit once three annotators give the prediction -- but scored against
    each leave-one-annotator-out subset in turn and averaged, so that an answer
    only three of ten people gave earns 0.9 rather than 1.0. Both sides go
    through [`normalize_answer`] first.

    An earlier version of this function compared raw strings against all ten
    annotators at once. That reads about 0.8 points high on VQA v1 val, and
    missing the normalization costs a little back; the two do not cancel.
    """
    if normalize:
        prediction = normalize_answer(prediction)
        human_answers = [normalize_answer(a) for a in human_answers]
    matches = [a == prediction for a in human_answers]
    total = sum(matches)
    # Leave one annotator out: that annotator's own vote is excluded.
    return float(np.mean([min((total - m) / 3.0, 1.0) for m in matches]))


class VQAMultipleChoiceDataset(Dataset):
    """One example per question: token ids, 18 candidate answers, image features.

    Args:
        vqa_h5_path: output of `scripts/prepare_vqa.py`.
        features_h5_path: id-indexed image features.
        split: `"train"` or `"val"`.
        in_memory: hold the feature table in RAM (~18 GB as float16).
        normalize_features: L2-normalize each region vector. Bottom-up features
            arrive with an L2 norm around 88 per box; feeding those through the
            encoder's `Linear -> Tanh` drives 98% of the activations past |0.99|,
            so the image path passes little more than sign information. The VGG
            and ResNet grids these models were built for are post-ReLU and far
            smaller, which is why the original needs no such step.
    """

    def __init__(
        self,
        vqa_h5_path: str,
        features_h5_path: str,
        split: str,
        in_memory: bool = False,
        normalize_features: bool = True,
    ):
        self.normalize_features = normalize_features
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

    @staticmethod
    def _l2_normalize(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / np.maximum(norms, 1e-6)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image = np.asarray(self._feature(self.feature_rows[index]), dtype=np.float32)
        if self.normalize_features:
            image = self._l2_normalize(image)
        return {
            "question_input_ids": self.questions[index].astype(np.int64),
            "choice_input_ids": self.choices[index].astype(np.int64),
            "image_features": image,
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
