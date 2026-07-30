"""Dataset for DSTC7 Audio-Visual Scene-Aware Dialog.

Reads what `scripts/prepare_avsd.py` writes. Video and audio features are stored
once per video and referenced by row, since every dialog has ten turns over one
video and storing them per turn would multiply the file tenfold.
"""

from typing import Any, Dict, Optional, Sequence

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
            # Which turn of the dialog this row is. Together with the video id it
            # names the row the way the DSTC7 references do -- `{video}_{turn}`.
            self.turn = h5[f"{split}_turn"][:]
            self.video = h5[f"{split}_video"][:] if in_memory else None
            self.audio = h5[f"{split}_audio"][:] if in_memory else None
            # Per-frame conv grids, written only when the h5 was prepared with
            # them. They are stored float16 and stay on disk regardless of
            # `in_memory`: frames x regions x channels per video is two orders of
            # magnitude above the pooled streams.
            self.has_frames = f"{split}_frames" in h5

    def __len__(self) -> int:
        return len(self.question)

    def _handle(self) -> h5py.File:
        if self._h5 is None:  # reopened per worker; handles do not survive a fork
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def _features(self, row: int):
        if self.video is not None:
            return self.video[row], self.audio[row]
        h5 = self._handle()
        return h5[f"{self.split}_video"][row], h5[f"{self.split}_audio"][row]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.feature_row[index]
        video, audio = self._features(row)
        item = {
            "question_input_ids": self.question[index].astype(np.int64),
            "history_input_ids": self.history[index].astype(np.int64),
            "answer_input_ids": self.answer[index].astype(np.int64),
            "video_features": np.asarray(video, dtype=np.float32),
            "audio_features": np.asarray(audio, dtype=np.float32),
        }
        if self.has_frames:
            item["frame_features"] = self._handle()[f"{self.split}_frames"][row].astype(np.float32)
        return item


class AVSDCollator:
    """Stack a list of examples into batched tensors.

    Args:
        zero: feature keys to blank out. A model can carry a modality without
            reading it, and the only way to tell the difference is to take it
            away: if the loss barely moves when the video is zeroed, the video
            was never being used, whatever the architecture diagram says.
    """

    def __init__(self, zero: Sequence[str] = ()):
        self.zero = tuple(zero)

    def __call__(self, features):
        batch = {
            key: torch.from_numpy(np.stack([np.asarray(f[key]) for f in features]))
            for key in features[0]
        }
        for key in self.zero:
            if key in batch:
                batch[key] = torch.zeros_like(batch[key])
        return batch


def load_vocab(path: str):
    """The word list, in id order; id 0 is padding and is not in the list.

    Id 1 is `<eos>`, so `words[0]` is that marker rather than a real word.
    """
    with h5py.File(path, "r") as h5:
        return [w.decode() if isinstance(w, bytes) else w for w in h5["vocab"][:]]


def render(ids, words, eos_token_id: int = 1) -> str:
    """Token ids as a sentence, stopping at `<eos>` and dropping padding.

    Everything a model emits after its end marker is an artefact of decoding to a
    fixed length -- included, it would be scored as if the model had written it.
    """
    kept = []
    for token in ids:
        token = int(token)
        if token == eos_token_id:
            break
        if token > 0:
            kept.append(words[token - 1])
    return " ".join(kept)
