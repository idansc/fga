"""Readers for pre-extracted image features.

FGA never looks at pixels: it consumes `(num_regions, feature_dim)` features per
image. Two containers are supported --

* an h5 file with `{split}_features`, which is what [`VisDialDataset`] expects and
  what `scripts/make_dev_image_features.py` writes;
* the bottom-up-attention LMDB published with the VisDial challenge starter code,
  which [`ImageFeaturesLmdbReader`] can convert into that h5 layout.
"""

import base64
import pickle
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["ImageFeaturesLmdbReader", "lmdb_to_h5"]


class ImageFeaturesLmdbReader:
    """Read bottom-up-attention features from an LMDB keyed by image id.

    Each record holds base64-encoded region features, boxes and class
    probabilities. A mean-pooled "whole image" region is prepended, so an image
    with 36 proposals yields 37 regions -- the number FGA's image utility expects.

    Args:
        features_path: path to the LMDB directory.
        in_memory: cache decoded records; the full dump is tens of GB, so leave
            this off unless you are reading a subset.
    """

    def __init__(self, features_path: str, in_memory: bool = False):
        import lmdb  # imported lazily: only needed for the LMDB path

        self.features_path = features_path
        self._in_memory = in_memory
        self.env = lmdb.open(features_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get(b"keys"))
        # A dict, not a list: the original did a linear `.index()` per lookup.
        self._index_of = {image_id: i for i, image_id in enumerate(self._image_ids)}
        self._cache: List[Optional[Tuple]] = [None] * len(self._image_ids)

    def __len__(self) -> int:
        return len(self._image_ids)

    def keys(self) -> List[bytes]:
        return self._image_ids

    def __getitem__(self, image_id) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
        """Returns `(features, num_boxes, boxes_normalized, boxes_absolute, class_probs)`."""
        key = str(image_id).encode() if not isinstance(image_id, bytes) else image_id
        index = self._index_of[key]
        if self._in_memory and self._cache[index] is not None:
            return self._cache[index]

        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(key))

        image_h, image_w = int(item["image_h"]), int(item["image_w"])
        num_boxes = int(item["num_boxes"])

        def decode(field: str, width: int) -> np.ndarray:
            return np.frombuffer(base64.b64decode(item[field]), dtype=np.float32).reshape(num_boxes, width)

        features = decode("features", 2048)
        boxes = decode("boxes", 4)
        class_probs = decode("cls_prob", 1601)

        # Region 0 stands for the whole image.
        global_probs = np.zeros(1601, dtype=np.float32)
        global_probs[0] = 1
        class_probs = np.concatenate([global_probs[None], class_probs], axis=0)
        features = np.concatenate([features.mean(axis=0, keepdims=True), features], axis=0)

        absolute = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        absolute[:, :4] = boxes
        absolute[:, 4] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) / float(image_w * image_h)

        normalized = absolute.copy()
        normalized[:, [0, 2]] /= float(image_w)
        normalized[:, [1, 3]] /= float(image_h)

        normalized = np.concatenate([np.array([[0, 0, 1, 1, 1]], dtype=np.float32), normalized], axis=0)
        absolute = np.concatenate(
            [np.array([[0, 0, image_w, image_h, image_w * image_h]], dtype=np.float32), absolute], axis=0
        )

        result = (features, num_boxes + 1, normalized, absolute, class_probs)
        if self._in_memory:
            self._cache[index] = result
        return result


def lmdb_to_h5(
    lmdb_path: str,
    h5_path: str,
    split: str,
    image_ids: List[int],
    num_regions: int = 37,
    feature_dim: int = 2048,
) -> None:
    """Convert an LMDB feature dump into the `{split}_features` h5 [`VisDialDataset`] reads.

    Args:
        lmdb_path: source LMDB directory.
        h5_path: destination h5; opened in append mode so splits can accumulate.
        split: `"train"`, `"val"` or `"test"`.
        image_ids: image ids in the order the split's dialogs use them, from
            `visdial_params.json` via `image_ids_from_params`.
        num_regions: regions to keep per image, padding or truncating as needed.
        feature_dim: feature width.
    """
    import h5py

    reader = ImageFeaturesLmdbReader(lmdb_path)
    with h5py.File(h5_path, "a") as h5:
        key = f"{split}_features"
        if key in h5:
            del h5[key]
        dataset = h5.create_dataset(key, shape=(len(image_ids), num_regions, feature_dim), dtype=np.float32)
        for i, image_id in enumerate(image_ids):
            features = reader[image_id][0]
            kept = min(num_regions, features.shape[0])
            dataset[i, :kept] = features[:kept]
            if i % 1000 == 0:
                print(f"{split}: {i}/{len(image_ids)}")
    print(f"Wrote {key} to {h5_path}")
