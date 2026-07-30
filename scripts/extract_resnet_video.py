#!/usr/bin/env python
"""Per-frame ResNet features for YouCook2, one `(frames, 2048)` array per video.

The retrieval reference loads `{video_id}_resnet.npy` from a hardcoded path on the
author's own machine, so the features cannot be downloaded and have to be
re-extracted. What matters for equivalence is the shape of the sequence rather
than the sampling rate: the loader maps a caption's timestamps onto feature rows
proportionally,

    start = floor(segment_start / duration * len(features))

so any constant frame rate lands the same window on the same part of the video. It
then subsamples anything longer than 80 rows, which puts a ceiling on how much a
higher rate could buy.

```bash
python scripts/extract_resnet_video.py --video_root retrieval/yc2_all/raw_videos \
    --output_dir retrieval/yc2_resnet --fps 1 --shard 0 --num_shards 8
```
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision


def frame_times(path, fps):
    """Timestamps to sample, and the capture handle, or `(None, None)` if unreadable."""
    capture = cv2.VideoCapture(path)
    rate = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not rate or rate != rate or total <= 0:  # nan or empty
        capture.release()
        return None, None
    duration = total / rate
    return np.arange(0.0, duration, 1.0 / fps), capture


def read_frames(path, fps, size=224):
    """Frames at `fps`, as a `(count, 3, size, size)` normalized tensor."""
    wanted, capture = frame_times(path, fps)
    if wanted is None:
        return None

    # Seeking per frame is slower than a single decode pass for these videos, so
    # step through and keep the frames whose index falls on a wanted timestamp.
    rate = capture.get(cv2.CAP_PROP_FPS)
    keep = {int(round(t * rate)) for t in wanted}
    frames, index = [], 0
    ok, image = capture.read()
    while ok:
        if index in keep:
            frames.append(cv2.resize(image, (size, size))[:, :, ::-1])
        ok, image = capture.read()
        index += 1
    capture.release()

    if not frames:
        return None
    batch = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (batch - mean) / std


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_root", required=True, help="Holds training/ validation/ testing/ subtrees.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    videos = []
    for root, _, files in os.walk(args.video_root):
        for name in files:
            if name.lower().endswith((".mp4", ".mkv", ".webm", ".avi")):
                videos.append(os.path.join(root, name))
    videos.sort()
    videos = videos[args.shard :: args.num_shards]
    print(f"shard {args.shard}: {len(videos)} videos", flush=True)

    # The classifier is dropped, so the output is the 2048-d pooled activation the
    # reference's `--img_dim` expects.
    weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet152(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.eval().cuda()

    written, unreadable = 0, 0
    for path in videos:
        video_id = os.path.splitext(os.path.basename(path))[0]
        target = os.path.join(args.output_dir, f"{video_id}.npy")
        if os.path.exists(target):
            continue

        frames = read_frames(path, args.fps)
        if frames is None:
            unreadable += 1
            continue

        features = []
        with torch.no_grad():
            for start in range(0, len(frames), args.batch_size):
                chunk = frames[start : start + args.batch_size].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features.append(model(chunk).float().half().cpu())
        np.save(target, torch.cat(features).numpy())
        written += 1
        if written % 25 == 0:
            print(f"shard {args.shard}: wrote {written}, {unreadable} unreadable", flush=True)

    print(f"shard {args.shard}: wrote {written}, {unreadable} unreadable", flush=True)


if __name__ == "__main__":
    sys.exit(main())
