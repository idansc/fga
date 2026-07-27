#!/usr/bin/env python
"""Build an id-indexed feature table for one VQA split.

`prepare_vqa.py` builds the train/val tables alongside the tokenized questions.
Test2015 has no answers, so it needs only the image side: this writes the same
`features` / `coco_ids` layout that `predict_vqa_test.py` reads.

```bash
python scripts/build_vqa_features.py \
    --features_dir vqa/test_npy \
    --questions vqa/raw/OpenEnded_mscoco_test2015_questions.json \
    --output vqa/test_features.h5
```
"""

import argparse
import json
import os
import re
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features_dir", required=True, help="Per-image .npy files named after the COCO images.")
    parser.add_argument("--questions", required=True, help="Which images are needed.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_regions", type=int, default=36)
    args = parser.parse_args()

    with open(args.questions) as handle:
        needed = sorted({q["image_id"] for q in json.load(handle)["questions"]})
    print(f"{len(needed)} images referenced")

    stem_of = {}
    for name in os.listdir(args.features_dir):
        if name.endswith(".npy") and not name.endswith("_info.npy"):
            match = re.search(r"(\d+)\.npy$", name)
            if match:
                stem_of[int(match.group(1))] = os.path.join(args.features_dir, name)

    missing = [i for i in needed if i not in stem_of]
    if missing:
        raise SystemExit(f"{len(missing)} images have no features, e.g. {missing[:5]}")

    sample = np.load(stem_of[needed[0]])
    with h5py.File(args.output, "w") as h5:
        table = h5.create_dataset(
            "features", (len(needed), args.num_regions, sample.shape[1]), dtype=np.float16
        )
        h5.create_dataset("coco_ids", data=np.asarray(needed, dtype=np.int64))
        for row, coco_id in enumerate(needed):
            feats = np.load(stem_of[coco_id])[: args.num_regions]
            table[row, : len(feats)] = feats.astype(np.float16)
            if row % 20000 == 0:
                print(f"  {row}/{len(needed)}", flush=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
