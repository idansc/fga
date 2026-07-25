#!/usr/bin/env python
"""Write a small random image-features h5, for exercising the pipeline without the real features.

The real F-RCNN dump is tens of gigabytes. This produces a file with the same
schema -- `{split}_features` of shape `(num_images, num_regions, feature_dim)` --
filled with noise, which is enough to check shapes, batching, metrics and the
submission format end to end. It says nothing about accuracy.

```bash
python scripts/make_dev_image_features.py --splits val --num_images 64 --output data/dev_features.h5
```
"""

import argparse

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/dev_features.h5")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--num_regions", type=int, default=37)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.output, "w") as h5:
        for split in args.splits:
            data = rng.standard_normal((args.num_images, args.num_regions, args.feature_dim), dtype=np.float32)
            h5.create_dataset(f"{split}_features", data=data, compression="gzip")
    print(f"Wrote {args.output}: {args.splits} x {args.num_images} images x {args.num_regions} x {args.feature_dim}")


if __name__ == "__main__":
    main()
