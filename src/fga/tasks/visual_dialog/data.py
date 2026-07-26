"""Data pipeline for VisDial v1.0.

The on-disk format is the one produced by the VisDial preprocessing scripts: a
single `visdial_data.h5` of integer-indexed dialogs plus a `visdial_params.json`
holding the vocabulary, both described in the README.

[`VisDialDataset`] yields dictionaries keyed exactly like the arguments of
[`FGAForVisualDialog.forward`], so a plain `default_data_collator` is enough;
[`VisDialCollator`] is provided for the padding-aware path and for turning numpy
into tensors without a copy through Python lists.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "VisDialDataset",
    "VisDialCollator",
    "vocab_size_from_params",
    "load_visdial_params",
    "build_hf_dataset",
]

#: Keys of every batch produced by [`VisDialDataset`].
FEATURE_KEYS = (
    "question_input_ids",
    "option_input_ids",
    "history_question_input_ids",
    "history_answer_input_ids",
    "caption_input_ids",
    "question_lengths",
    "option_lengths",
    "caption_lengths",
    "image_features",
)


def load_visdial_params(path: str) -> Dict[str, Any]:
    """Load `visdial_params.json`, validating that it carries a vocabulary."""
    with open(path, "r") as handle:
        params = json.load(handle)
    if "word2ind" not in params:
        raise ValueError(f"{path} has no 'word2ind' dictionary; it is not a VisDial params file.")
    return params


def vocab_size_from_params(params: Dict[str, Any]) -> int:
    """Number of embedding rows implied by a params file.

    That is `len(word2ind) + 3`: id 0 is padding, ids `1..len(word2ind)` are real
    words, and the pipeline appends a `<stop>` and an `<empty>` symbol.
    """
    return len(params["word2ind"]) + 3


class VisDialDataset(Dataset):
    """One example per dialog round: a question, 100 candidate answers, and context.

    Args:
        visdial_data_path: path to `visdial_data.h5`.
        image_features_path: path to an h5 with a `{split}_features` dataset of
            shape `(num_images, num_regions, feature_dim)`.
        split: `"train"`, `"val"` or `"test"`.
        vocab_size: embedding table size, see [`vocab_size_from_params`]. Used to
            derive the `<stop>` and `<empty>` ids.
        trunc_length: number of tokens questions and answers are truncated to.
        caption_length: number of tokens captions are truncated to.
        add_stop_to_answers: append a `<stop>` symbol after the last answer word.
        add_stop_to_questions: append a `<stop>` symbol after the last question
            and caption word.
        limit_images: keep only the first N images; useful for smoke tests.
        normalize_image_features: L2-normalize each image's features. The norm is
            taken over all regions jointly, matching the released checkpoints.
        in_memory: read the h5 datasets into RAM up front. The image features
            dominate; set `False` to stream them from disk instead.
    """

    def __init__(
        self,
        visdial_data_path: str,
        image_features_path: str,
        split: str,
        vocab_size: int,
        trunc_length: int = 20,
        caption_length: int = 40,
        add_stop_to_answers: bool = True,
        add_stop_to_questions: bool = True,
        limit_images: Optional[int] = None,
        normalize_image_features: bool = True,
        in_memory: bool = True,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.split = split
        self.trunc_length = int(trunc_length)
        self.caption_length = int(caption_length)
        self.add_stop_to_answers = add_stop_to_answers
        self.add_stop_to_questions = add_stop_to_questions
        self.stop_id = vocab_size - 2
        self.empty_id = vocab_size - 1
        self.image_features_path = image_features_path
        self._in_memory = in_memory
        self._image_handle: Optional[h5py.File] = None

        limit = limit_images
        with h5py.File(visdial_data_path, "r") as h5:
            self.ques = h5[f"ques_{split}"][:limit]
            # Rounds per dialog: 10 for VisDial, 9 for the VisDial-Q variant.
            self.n_qa_per_dial = self.ques.shape[1]
            self.ques = self.ques.reshape(-1, self.trunc_length)
            self.ques_length = h5[f"ques_length_{split}"][:limit].reshape(-1)

            self.ans = h5[f"ans_{split}"][:limit].reshape(-1, self.trunc_length)
            self.ans_length = h5[f"ans_length_{split}"][:limit].reshape(-1)

            # Options are stored 1-indexed into the answer pool; shift to 0-indexed.
            self.opt = h5[f"opt_{split}"][:limit].reshape(-1, 100) - 1

            self.ans_index = None
            if split != "test":
                self.ans_index = h5[f"ans_index_{split}"][:limit].reshape(-1) - 1

            self.opt_list = h5[f"opt_list_{split}"][:]
            self.opt_length_list = h5[f"opt_length_{split}"][:]

            self.cap = h5[f"cap_{split}"][:limit]
            self.cap_length = h5[f"cap_length_{split}"][:limit]

            # How many rounds each dialog really has. Test dialogs are cut at a
            # random round, and only that round is scored by the challenge server.
            rounds_key = f"num_rounds_{split}"
            self.num_rounds_per_image = (
                h5[rounds_key][:limit] if rounds_key in h5 else np.full(len(self.cap), self.n_qa_per_dial)
            )

        with h5py.File(image_features_path, "r") as img_h5:
            key = f"{split}_features"
            if key not in img_h5:
                raise KeyError(
                    f"{image_features_path} has no dataset '{key}'. Available: {sorted(img_h5.keys())}. "
                    "Grid (VGG) feature dumps use different key names and need renaming."
                )
            self.image_feature_shape = img_h5[key].shape[1:]
            num_feature_rows = img_h5[key].shape[0]
            num_images = len(self.cap)
            if num_feature_rows < num_images:
                raise ValueError(
                    f"{image_features_path}['{key}'] has {num_feature_rows} images but the {split} split has "
                    f"{num_images} dialogs. Pass limit_images<={num_feature_rows} to work with a feature subset, "
                    "or extract features for the whole split."
                )
            if in_memory:
                images = img_h5[key][:limit]
                self.images = self._normalize(images) if normalize_image_features else images
            else:
                self.images = None
                self._num_images = img_h5[key].shape[0] if limit is None else min(limit, img_h5[key].shape[0])
        self.normalize_image_features = normalize_image_features

        self.total_samples = len(self.ques)
        assert self.total_samples == len(self.ans) == len(self.opt) == len(self.ques_length) == len(self.ans_length), (
            "Dialog arrays disagree on the number of rounds."
        )

    @staticmethod
    def _normalize(images: np.ndarray) -> np.ndarray:
        """L2-normalize each image over all regions jointly, as in the paper code."""
        shape = images.shape
        flat = images.reshape(shape[0], -1).astype(np.float32)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        np.divide(flat, np.maximum(norms, 1e-12), out=flat)
        return flat.reshape(shape)

    def _get_image(self, img_index: int) -> np.ndarray:
        if self.images is not None:
            return self.images[img_index]
        # Lazily reopen per worker process; h5py handles do not survive a fork.
        if self._image_handle is None:
            self._image_handle = h5py.File(self.image_features_path, "r", swmr=True)
        features = self._image_handle[f"{self.split}_features"][img_index]
        if self.normalize_image_features:
            return self._normalize(features[None])[0]
        return features

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        trunc = self.trunc_length
        option_ids = self.opt_list[self.opt[index]].astype(np.int64)
        option_lengths = self.opt_length_list[self.opt[index]].astype(np.int64)
        if self.add_stop_to_answers:
            option_ids = np.insert(option_ids, trunc, 0, axis=1)
            option_lengths = option_lengths + 1
            option_ids[np.arange(option_ids.shape[0]), option_lengths - 1] = self.stop_id

        question_ids = self.ques[index].astype(np.int64)
        question_length = np.int64(self.ques_length[index])
        if self.add_stop_to_questions:
            question_ids = np.insert(question_ids, trunc, 0, axis=0)
            question_length = question_length + 1
            question_ids[question_length - 1] = self.stop_id

        # History: one row per previous round, `<empty> <stop>` where nothing happened yet.
        history_len = trunc + 1
        n_rounds = self.n_qa_per_dial - 1
        hist_ques = np.zeros((n_rounds, history_len), dtype=np.int64)
        hist_ans = np.zeros((n_rounds, history_len), dtype=np.int64)
        hist_ques[:, 0] = self.empty_id
        hist_ans[:, 0] = self.empty_id
        hist_ques[:, 1] = self.stop_id
        hist_ans[:, 1] = self.stop_id

        dialog_start = (index // self.n_qa_per_dial) * self.n_qa_per_dial
        past = np.arange(dialog_start, index)
        qhist = np.insert(self.ques[past], trunc, 0, axis=1)
        ahist = np.insert(self.ans[past], trunc, 0, axis=1)
        n_past = len(past)
        qhist[np.arange(n_past), self.ques_length[past]] = self.stop_id
        ahist[np.arange(n_past), self.ans_length[past]] = self.stop_id
        hist_ques[:n_past] = qhist
        hist_ans[:n_past] = ahist

        img_index = index // self.n_qa_per_dial
        caption_ids = self.cap[img_index].astype(np.int64)
        caption_length = np.int64(self.cap_length[img_index])
        if self.add_stop_to_questions:
            caption_ids = np.insert(caption_ids, self.caption_length, 0)
            caption_length = caption_length + 1
            caption_ids[caption_length - 1] = self.stop_id

        example = {
            "question_input_ids": question_ids,
            "option_input_ids": option_ids,
            "history_question_input_ids": hist_ques,
            "history_answer_input_ids": hist_ans,
            "caption_input_ids": caption_ids,
            "question_lengths": question_length,
            "option_lengths": option_lengths,
            "caption_lengths": caption_length,
            "image_features": np.asarray(self._get_image(img_index), dtype=np.float32),
        }
        if self.ans_index is not None:
            example["labels"] = np.int64(self.ans_index[index])
        return example


@dataclass
class VisDialCollator:
    """Stacks examples into batched tensors.

    Args:
        return_labels: drop `labels` when `False`, e.g. for test-set inference.
    """

    return_labels: bool = True

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        keys = list(FEATURE_KEYS)
        if self.return_labels and "labels" in features[0]:
            keys.append("labels")

        batch = {}
        for key in keys:
            stacked = np.stack([np.asarray(example[key]) for example in features])
            batch[key] = torch.from_numpy(stacked)
        return batch


def build_hf_dataset(dataset: VisDialDataset, num_proc: Optional[int] = None, cache_dir: Optional[str] = None):
    """Wrap a [`VisDialDataset`] as a `datasets.Dataset`.

    Optional -- [`transformers.Trainer`] takes the torch dataset directly, and
    materializing 1.2M rounds of 100 candidate answers each costs a great deal of
    disk. Use this when you want `datasets` features such as `map`, streaming to
    the Hub, or memory-mapped sharing between workers.
    """
    from datasets import Dataset as HFDataset

    def generator():
        for i in range(len(dataset)):
            yield dataset[i]

    return HFDataset.from_generator(
        generator,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )


class DenseAnnotationsReader:
    """Random access to the VisDial v1.0 val dense relevance annotations.

    The file must follow the schema published at https://visualdialog.org/data.

    Args:
        dense_annotations_jsonpath: path to `visdial_1.0_val_dense_annotations.json`.
    """

    def __init__(self, dense_annotations_jsonpath: str):
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
        # A dict lookup; the original code did a linear `list.index` per image,
        # which made evaluation quadratic in the size of the val split.
        self._by_image_id = {entry["image_id"]: entry for entry in self._visdial_data}

    def __len__(self) -> int:
        return len(self._by_image_id)

    def __contains__(self, image_id: int) -> bool:
        return image_id in self._by_image_id

    def __getitem__(self, image_id: int) -> Dict[str, Any]:
        # keys: {"image_id", "round_id", "gt_relevance"}
        return self._by_image_id[image_id]

    @property
    def split(self) -> str:
        return "val"


def image_ids_from_params(params: Dict[str, Any], split: str) -> List[int]:
    """COCO/VisDial image ids for a split, parsed from the stored filenames."""
    key = f"unique_img_{split}"
    if key not in params:
        raise KeyError(f"params file has no '{key}'; available: {sorted(k for k in params if k.startswith('unique'))}")
    return [int(os.path.basename(name)[-16:-4]) for name in params[key]]
