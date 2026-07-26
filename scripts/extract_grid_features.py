#!/usr/bin/env python
"""Extract a 14x14 ResNet grid over a directory of images, into one h5.

The VQA models attend over a spatial grid rather than detected regions: 196
cells of 2048 channels, from the last convolutional stage of a ResNet-152 at
448x448 input. This walks a directory, runs the backbone, and writes

    {split}_features  (num_images, 196, 2048)   float16
    {split}_ids       (num_images,)             int64   COCO ids, same order

Rows follow sorted filename order; the ids dataset is what training joins
against, so nothing depends on the order beyond the pairing.

```bash
python scripts/extract_grid_features.py \
    --image_dirs data/images/train2014 data/images/val2014 \
    --split trainval --output vqa/grid_features.h5 --batch_size 96
```
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ImageFolder(Dataset):
    """Images from one or more directories, in sorted filename order."""

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, image_dirs, resolution: int):
        self.paths = sorted(
            os.path.join(d, name)
            for d in image_dirs
            for name in os.listdir(d)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def coco_id(path: str) -> int:
        """`COCO_train2014_000000378466.jpg` -> 378466."""
        match = re.search(r"(\d+)\.\w+$", os.path.basename(path))
        return int(match.group(1))

    def __getitem__(self, index):
        from PIL import Image
        from torchvision.transforms import functional as TF

        image = Image.open(self.paths[index]).convert("RGB")
        tensor = TF.to_tensor(TF.resize(image, (self.resolution, self.resolution)))
        return (tensor - self.MEAN) / self.STD, self.coco_id(self.paths[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dirs", nargs="+", required=True)
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=448, help="448 -> 14x14 grid from a ResNet.")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    import torchvision

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    # Everything up to the final conv stage; drops the pooling and classifier.
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2]).to(device).eval()

    dataset = ImageFolder(args.image_dirs, args.resolution)
    grid = args.resolution // 32
    print(f"{len(dataset)} images -> ({grid}x{grid})x2048 on {device}")

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "a") as h5:
        for key in (f"{args.split}_features", f"{args.split}_ids"):
            if key in h5:
                del h5[key]
        features = h5.create_dataset(f"{args.split}_features", (len(dataset), grid * grid, 2048), dtype=np.float16)
        ids = h5.create_dataset(f"{args.split}_ids", (len(dataset),), dtype=np.int64)

        done = 0
        with torch.inference_mode():
            for batch, batch_ids in loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = backbone(batch.to(device, non_blocking=True))
                out = out.flatten(2).transpose(1, 2)  # (batch, grid*grid, 2048)
                features[done : done + out.size(0)] = out.float().cpu().numpy().astype(np.float16)
                ids[done : done + out.size(0)] = batch_ids.numpy()
                done += out.size(0)
                if done % (args.batch_size * 50) < args.batch_size:
                    print(f"  {done}/{len(dataset)}", flush=True)

    print(f"wrote {args.output}::{args.split}_features {done} rows")


if __name__ == "__main__":
    sys.exit(main())
