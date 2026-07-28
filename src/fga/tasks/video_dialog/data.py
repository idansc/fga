"""Dataset for DSTC7 Audio-Visual Scene-Aware Dialog.

Reads what `scripts/prepare_avsd.py` writes. Video and audio features are stored
once per video and referenced by row, since every dialog has ten turns over one
video and storing them per turn would multiply the file tenfold.
"""

from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["AVSDDataset", "AVSDCollator", "load_vocab"]


class AVSDDataset(Dataset):
    """One example per dialog turn.

    Args:
        path: the h5 from `scripts/prepare_avsd.py`.
        split: `"train"` or `"val"`.
        in_memory: hold the video and audio tables in RAM. The video table is the
            large one -- streams x segments x 2048 per video -- so this is worth
            checking against available memory before enabling.
    """

    def __init__(self, path: str, split: str, in_memory: bool = False):
        self.path = path
        self.split = split
        self._h5: Optional[h5py.File] = None

        with h5py.File(path, "r") as h5:
            self.question = h5[f"{split}_question"][:]
            self.answer = h5[f"{split}_answer"][:]
            self.history = h5[f"{split}_history"][:]
            self.feature_row = h5[f"{split}_feature_row"][:]
            self.video_id = h5[f"{split}_video_id"][:]
            self.video = h5[f"{split}_video"][:] if in_memory else None
            self.audio = h5[f"{split}_audio"][:] if in_memory else None

    def __len__(self) -> int:
        return len(self.question)

    def _features(self, row: int):
        if self.video is not None:
            return self.video[row], self.audio[row]
        if self._h5 is None:  # reopened per worker; handles do not survive a fork
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5[f"{self.split}_video"][row], self._h5[f"{self.split}_audio"][row]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video, audio = self._features(self.feature_row[index])
        return {
            "question_input_ids": self.question[index].astype(np.int64),
            "history_input_ids": self.history[index].astype(np.int64),
            "answer_input_ids": self.answer[index].astype(np.int64),
            "video_features": np.asarray(video, dtype=np.float32),
            "audio_features": np.asarray(audio, dtype=np.float32),
        }


class AVSDCollator:
    def __call__(self, features):
        return {
            key: torch.from_numpy(np.stack([np.asarray(f[key]) for f in features]))
            for key in features[0]
        }


def load_vocab(path: str):
    """The word list, in id order; id 0 is padding and is not in the list."""
    with h5py.File(path, "r") as h5:
        return [w.decode() if isinstance(w, bytes) else w for w in h5["vocab"][:]]
