#!/usr/bin/env python
"""Build a text-to-video retrieval set from YouCook2.

VideoMatch was published on DiDeMo and ActivityNet Captions, whose pre-extracted
features the paper distributes through a Google Drive folder that now returns 404.
YouCook2 is a stand-in of the same shape -- a video, timestamped segments, and a
sentence per segment -- with InternVideo clip features available on the Hub.

Two things this is *not*. It is not the paper's benchmark, so the numbers do not
compare to its published ones. And the feature dump covers YouCook2's validation
videos only, so the split here is over those 436 videos rather than the official
train/val division; it is held out by *video*, so no video appears on both sides.

Each captioned segment becomes one example: the clip features overlapping its
timestamps, and the sentence as GloVe vectors.

```bash
python scripts/prepare_youcook.py --features_dir yc2/internvideo_MM_L14_features \
    --annotations yc2/YouCookII/annotations/youcookii_annotations_trainval.json \
    --glove glove/glove.6B.300d.txt --output yc2/retrieval.h5
```
"""

import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import torch

TOKEN = re.compile(r"[a-z0-9']+")


def load_glove(path, needed):
    """Only the words that occur, so the 1 GB file is read once and dropped."""
    vectors = {}
    with open(path, encoding="utf8") as handle:
        for line in handle:
            word, _, rest = line.partition(" ")
            if word in needed:
                vectors[word] = np.fromstring(rest, sep=" ", dtype=np.float32)
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--glove", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_clips", type=int, default=32, help="Clips kept per segment.")
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.annotations) as handle:
        database = json.load(handle)["database"]
    features = {
        os.path.basename(p).replace(".pth.tar", ""): p
        for p in glob.glob(os.path.join(args.features_dir, "*", "*.pth.tar"))
    }
    videos = sorted(set(features) & set(database))
    print(f"{len(videos)} videos with both features and annotations")

    words = set()
    for video in videos:
        for segment in database[video]["annotations"]:
            words.update(TOKEN.findall(segment["sentence"].lower()))
    glove = load_glove(args.glove, words)
    print(f"{len(words)} distinct words, {len(glove)} found in GloVe")

    # Held out by video: a caption whose video was seen in training would be a
    # much easier retrieval problem than the task intends.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(videos))
    cut = int(len(videos) * (1 - args.test_fraction))
    split_of = {videos[i]: ("train" if rank < cut else "test") for rank, i in enumerate(order)}

    rows = {"train": [], "test": []}
    for video in videos:
        clips = torch.load(features[video], map_location="cpu", weights_only=False).float().numpy()
        duration = database[video]["duration"]
        per_clip = duration / max(len(clips), 1)

        for segment in database[video]["annotations"]:
            start, end = segment["segment"]
            first = int(start / per_clip)
            last = max(first + 1, int(np.ceil(end / per_clip)))
            window = clips[first:last][: args.max_clips]
            if len(window) == 0:
                continue

            video_features = np.zeros((args.max_clips, clips.shape[1]), dtype=np.float32)
            video_features[: len(window)] = window

            tokens = [w for w in TOKEN.findall(segment["sentence"].lower()) if w in glove][: args.max_words]
            if not tokens:
                continue
            text_features = np.zeros((args.max_words, 300), dtype=np.float32)
            for i, word in enumerate(tokens):
                text_features[i] = glove[word]

            rows[split_of[video]].append((video_features, text_features, len(window), len(tokens), video))

    with h5py.File(args.output, "w") as h5:
        for split, items in rows.items():
            if not items:
                continue
            video_features, text_features, clip_counts, word_counts, ids = zip(*items)
            h5.create_dataset(f"{split}_video", data=np.stack(video_features), dtype=np.float32)
            h5.create_dataset(f"{split}_text", data=np.stack(text_features), dtype=np.float32)
            h5.create_dataset(f"{split}_num_clips", data=np.asarray(clip_counts, dtype=np.int32))
            h5.create_dataset(f"{split}_num_words", data=np.asarray(word_counts, dtype=np.int32))
            h5.create_dataset(f"{split}_video_id", data=np.array(ids, dtype=h5py.string_dtype()))
            print(f"{split}: {len(items)} segments over {len(set(ids))} videos")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
