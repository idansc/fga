#!/usr/bin/env python
"""Extract per-frame spatial features from videos, keeping the grid.

The I3D features DSTC7 ships are global-average-pooled over height and width, so
"which region of this frame" has nothing to attend over. This keeps the conv grid:
VGG19's final pooling gives 7x7 = 49 regions of 512 channels per frame, which is
what `AVSDConfig.num_video_regions` already expects, and what lets each frame be
declared as its own modality with attention running inside it.

One `.npy` per video, shaped `(frames, 49, 512)`, float16. Frames are sampled
uniformly across the whole video rather than from the start, so a fixed count
covers long and short clips alike.

Shard across GPUs with `--shard i --num_shards n`; re-running skips videos whose
output already exists, so an interrupted shard resumes.

```bash
python scripts/extract_spatial_frames.py --video_dir charades/Charades_v1 \
    --output_dir charades_spatial --frames 48
```
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision


def sample_frames(path, count, size=224):
    """`count` frames spread across the video, as a `(count, 3, size, size)` tensor."""
    capture = cv2.VideoCapture(path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return None

    wanted = np.linspace(0, max(total - 1, 0), count).astype(int)
    frames, index, taken = [], 0, 0
    ok, image = capture.read()
    while ok and taken < count:
        while taken < count and wanted[taken] == index:
            frames.append(cv2.resize(image, (size, size))[:, :, ::-1])
            taken += 1
        ok, image = capture.read()
        index += 1
    capture.release()

    if not frames:
        return None
    while len(frames) < count:  # short or truncated video: repeat the last frame
        frames.append(frames[-1])

    batch = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (batch - mean) / std


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=48, help="Frames per forward pass.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # `features` ends at the final max-pool, so a 224 input leaves 512 x 7 x 7.
    model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    model = model.features.eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    videos = sorted(v for v in os.listdir(args.video_dir) if v.endswith((".mp4", ".mkv", ".webm")))
    videos = videos[args.shard :: args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(videos)} videos", flush=True)

    done = failed = 0
    for position, name in enumerate(videos):
        stem = os.path.splitext(name)[0]
        output = os.path.join(args.output_dir, f"{stem}.npy")
        if os.path.exists(output):  # resume rather than redo
            continue

        batch = sample_frames(os.path.join(args.video_dir, name), args.frames)
        if batch is None:
            failed += 1
            continue

        grids = []
        with torch.no_grad():
            for start in range(0, len(batch), args.batch_size):
                chunk = batch[start : start + args.batch_size].to(args.device)
                grid = model(chunk)                       # (n, 512, 7, 7)
                grids.append(grid.flatten(2).transpose(1, 2).half().cpu())  # (n, 49, 512)
        np.save(output, torch.cat(grids).numpy())
        done += 1
        if position % 200 == 0:
            print(f"  {position}/{len(videos)}  written {done}  unreadable {failed}", flush=True)

    print(f"shard {args.shard}: wrote {done}, {failed} unreadable")


if __name__ == "__main__":
    sys.exit(main())
