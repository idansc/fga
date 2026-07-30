#!/usr/bin/env python
"""Build a text-to-video retrieval set from YouCook2.

Each captioned segment becomes one example: the video features overlapping its
timestamps, and the sentence as GloVe vectors.

The split comes from YouCook2's own `subset` field -- 1,333 training videos with
10,337 segments, 457 validation videos with 3,492 -- which is what the retrieval
literature reports on and what
[Video and Text Matching with Conditioned Embeddings](https://arxiv.org/abs/2110.11298)
uses. Its loader keys off the same field.

`--features_dir` takes either layout:

* `.npy` per video, from `scripts/extract_resnet_video.py`. The reference loads
  `{video_id}_resnet.npy` from a path on the author's own machine, so these have
  to be re-extracted; ResNet-152 at 1 fps reproduces the shape it expects.
* `.pth.tar` under a per-recipe subdirectory, which is how the InternVideo dump on
  the Hub is arranged. That dump covers the *validation* videos only, so it cannot
  build the official split -- passing it here will simply leave the training half
  nearly empty, which the printed counts will show.

A segment's window is located proportionally, `start / duration * len(features)`,
the way the reference does it, so the frame rate the features were extracted at
does not have to be known here.

```bash
python scripts/prepare_youcook.py --features_dir yc2_resnet \
    --annotations yc2/YouCookII/annotations/youcookii_annotations_trainval.json \
    --glove glove/glove.6B.300d.txt --output yc2/retrieval_official.h5
```
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

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
    args = parser.parse_args()

    with open(args.annotations) as handle:
        database = json.load(handle)["database"]
    features = {
        os.path.basename(p).replace(".pth.tar", "").replace(".npy", ""): p
        for pattern in ("*.npy", os.path.join("*", "*.pth.tar"))
        for p in glob.glob(os.path.join(args.features_dir, pattern))
    }
    videos = sorted(set(features) & set(database))
    print(f"{len(videos)} videos with both features and annotations")

    have = Counter(database[v]["subset"] for v in videos)
    want = Counter(v["subset"] for v in database.values())
    for subset in ("training", "validation"):
        print(f"  {subset}: {have[subset]} of {want[subset]} videos have features")
    if have["training"] < 0.5 * want["training"]:
        print(
            "  warning: most training videos have no features, so this is not the "
            "official split -- see the module docstring."
        )

    words = set()
    for video in videos:
        for segment in database[video]["annotations"]:
            words.update(TOKEN.findall(segment["sentence"].lower()))
    glove = load_glove(args.glove, words)
    print(f"{len(words)} distinct words, {len(glove)} found in GloVe")

    # YouCook2's own division. `test` is this repo's name for the reported split,
    # which for YouCook2 retrieval is the validation subset -- the test subset's
    # annotations were never released.
    split_of = {
        video: ("train" if database[video]["subset"] == "training" else "test") for video in videos
    }

    rows = {"train": [], "test": []}
    for video in videos:
        path = features[video]
        clips = (
            np.load(path).astype(np.float32)
            if path.endswith(".npy")
            else torch.load(path, map_location="cpu", weights_only=False).float().numpy()
        )
        clips = clips.reshape(len(clips), -1)
        duration = database[video]["duration"]

        for segment in database[video]["annotations"]:
            start, end = segment["segment"]
            # Proportional, as the reference does it, so the extraction frame rate
            # never has to be known: a row index is a fraction of the way through.
            first = int(np.floor(start / duration * len(clips)))
            last = max(first + 1, int(np.ceil(end / duration * len(clips))) + 1)
            window = clips[first:last]
            if len(window) > args.max_clips:
                # Even subsampling rather than the head, so a long segment is
                # summarized instead of clipped to its opening seconds.
                window = window[np.linspace(0, len(window) - 1, args.max_clips).astype(int)]
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
