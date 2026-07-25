#!/usr/bin/env python
"""Assemble extracted region features into the `{split}_features` h5 FGA reads.

Feature extractors emit one file per image, named after the image. FGA instead
indexes images by *position*: `img_index = round_index // num_rounds`, where the
order is the one in `visdial_params.json["unique_img_{split}"]`. Getting that
mapping wrong does not crash anything -- it silently pairs every dialog with the
wrong picture -- so this script does the ordering explicitly and refuses to write
a file with gaps unless told otherwise.

Supported inputs:

* `--source npy` -- a directory of `<image_stem>.npy` arrays of shape
  `(num_boxes, feature_dim)`, which is what the MMF / vqa-maskrcnn-benchmark
  extractor (`extract_features_vmb.py`) produces.
* `--source lmdb` -- the bottom-up-attention LMDB layout.

The paper's image utility expects 37 regions: 36 proposals plus one mean-pooled
region standing for the whole image, prepended. That global region is added here
unless `--no_global_region` is passed.

```bash
python scripts/build_features_h5.py \
    --source npy --features_dir features/val --split val \
    --output data/frcnn_features_new.h5
```

Splits accumulate, so run it once per split against the same `--output`.
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.data import load_visdial_params  # noqa: E402


def image_stems(params: dict, split: str) -> list:
    """Image identifiers for a split, in the order FGA indexes them."""
    key = f"unique_img_{split}"
    if key not in params:
        raise KeyError(f"params file has no '{key}'")
    # e.g. "VisualDialog_val2018/VisualDialog_val2018_000000185565.jpg"
    return [os.path.splitext(os.path.basename(name))[0] for name in params[key]]


def build_index(features_dir: str) -> dict:
    """Map image stem -> path, ignoring the extractor's `_info` sidecar files."""
    index = {}
    for entry in os.scandir(features_dir):
        if not entry.name.endswith(".npy") or entry.name.endswith("_info.npy"):
            continue
        index[os.path.splitext(entry.name)[0]] = entry.path
    return index


def prepare(features: np.ndarray, num_regions: int, add_global: bool) -> np.ndarray:
    """Prepend the mean-pooled global region, then pad or truncate to `num_regions`."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"expected (num_boxes, feature_dim), got {features.shape}")
    if add_global:
        features = np.concatenate([features.mean(axis=0, keepdims=True), features], axis=0)

    if features.shape[0] >= num_regions:
        return features[:num_regions]
    padded = np.zeros((num_regions, features.shape[1]), dtype=np.float32)
    padded[: features.shape[0]] = features
    return padded


def infer_feature_dim(index: dict, stems: list) -> int:
    for stem in stems:
        if stem in index:
            return int(np.load(index[stem]).shape[1])
    raise SystemExit("Could not read any feature file to infer the feature dimension.")


def build_from_npy(args, stems: list) -> None:
    index = build_index(args.features_dir)
    print(f"{len(index)} feature files in {args.features_dir}; split needs {len(stems)}")

    missing = [stem for stem in stems if stem not in index]
    if missing:
        print(f"\nMissing features for {len(missing)}/{len(stems)} images, e.g.:")
        for stem in missing[:5]:
            print(f"  {stem}")
        if not args.allow_missing:
            raise SystemExit(
                "\nRefusing to write a features file with gaps: the missing rows would be "
                "silently paired with real dialogs. Re-run the extractor for these images, "
                "or pass --allow_missing to zero-fill them deliberately."
            )
        print("--allow_missing: these rows will be zero-filled.\n")

    feature_dim = args.feature_dim or infer_feature_dim(index, stems)
    print(f"writing {len(stems)} x {args.num_regions} x {feature_dim} to {args.output}::{args.split}_features")

    with h5py.File(args.output, "a") as h5:
        key = f"{args.split}_features"
        if key in h5:
            if not args.overwrite:
                raise SystemExit(f"{key} already exists in {args.output}; pass --overwrite to replace it.")
            del h5[key]
        dataset = h5.create_dataset(
            key,
            shape=(len(stems), args.num_regions, feature_dim),
            dtype=np.float32,
            compression="gzip" if args.compress else None,
        )
        for i, stem in enumerate(stems):
            if stem in index:
                dataset[i] = prepare(np.load(index[stem]), args.num_regions, not args.no_global_region)
            if i and i % 5000 == 0:
                print(f"  {i}/{len(stems)}")

    print(f"done: {args.output}::{key}")


def build_from_lmdb(args, stems: list) -> None:
    from fga.data import image_ids_from_params
    from fga.image_features import lmdb_to_h5

    params = load_visdial_params(args.visdial_params_path)
    lmdb_to_h5(
        lmdb_path=args.features_dir,
        h5_path=args.output,
        split=args.split,
        image_ids=image_ids_from_params(params, args.split),
        num_regions=args.num_regions,
        feature_dim=args.feature_dim or 2048,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features_dir", required=True, help="Directory of per-image .npy files, or an LMDB path.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", default="data/frcnn_features_new.h5")
    parser.add_argument("--visdial_params_path", default="data/visdial_params.json")
    parser.add_argument("--source", choices=["npy", "lmdb"], default="npy")
    parser.add_argument("--num_regions", type=int, default=37, help="36 proposals + 1 global region.")
    parser.add_argument("--feature_dim", type=int, default=None, help="Inferred from the files when omitted.")
    parser.add_argument("--no_global_region", action="store_true", help="Do not prepend the mean-pooled region.")
    parser.add_argument("--allow_missing", action="store_true", help="Zero-fill images with no features.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress", action="store_true", help="gzip the dataset; smaller but slower to read.")
    args = parser.parse_args()

    params = load_visdial_params(args.visdial_params_path)
    stems = image_stems(params, args.split)

    if args.source == "lmdb":
        build_from_lmdb(args, stems)
    else:
        build_from_npy(args, stems)


if __name__ == "__main__":
    main()
