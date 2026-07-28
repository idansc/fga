#!/usr/bin/env python
"""Prepare DSTC7 Audio-Visual Scene-Aware Dialog.

Turns the challenge's dialog JSONs and per-video feature files into one h5 the
dataset reads. Each of the ten turns in a dialog becomes an example: the question,
the turns before it as history, and the video's features.

The features DSTC7 distributes are pooled over space -- `i3d_rgb` is
`(segments, 2048)`, not `(segments, regions, 2048)` -- so the entities here are
*moments*, and attention answers "when in this video" rather than "where in this
frame". Per-frame spatial features need conv maps extracted from the Charades
videos; the layer takes them without modification once they exist, since frames
can be declared as weight-sharing modalities the way dialog rounds are.

Variable-length streams are padded or truncated to a fixed count, which the
learned marginalization in the pairwise factors requires.

Features are stored once per video with an index per turn, not once per turn: a
video's padded streams are ~786 KB and every dialog has ten turns, so the naive
layout writes the same array ten times and the file runs to tens of gigabytes.

```bash
python scripts/prepare_avsd.py --dstc_dir avsd/dstc7 --output avsd/avsd.h5
```
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import h5py
import numpy as np

TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def fixed(array, count):
    """Pad with zeros or truncate to exactly `count` rows."""
    out = np.zeros((count, array.shape[1]), dtype=np.float32)
    take = min(len(array), count)
    out[:take] = array[:take]
    return out, take


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dstc_dir", required=True, help="Holds the dialog JSONs and feats/.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video_segments", type=int, default=48)
    parser.add_argument("--audio_segments", type=int, default=48)
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--max_history", type=int, default=60, help="Words of history kept, most recent first.")
    parser.add_argument("--min_count", type=int, default=2, help="Word frequency floor, from train only.")
    args = parser.parse_args()

    splits = {}
    for name, filename in (
        ("train", "train_set4DSTC7-AVSD.json"),
        ("val", "valid_set4DSTC7-AVSD.json"),
    ):
        with open(os.path.join(args.dstc_dir, filename)) as handle:
            splits[name] = json.load(handle)["dialogs"]
        print(f"{name}: {len(splits[name])} dialogs")

    # Vocabulary from train only, over questions, answers and captions alike --
    # the answer decoder and the question encoder share it.
    counts = Counter()
    for dialog in splits["train"]:
        counts.update(tokenize(dialog["caption"]))
        counts.update(tokenize(dialog["summary"]))
        for turn in dialog["dialog"]:
            counts.update(tokenize(turn["question"]))
            counts.update(tokenize(turn["answer"]))
    words = [w for w, c in counts.most_common() if c >= args.min_count]
    word_to_id = {w: i + 1 for i, w in enumerate(words)}  # 0 is padding
    print(f"{len(words)} words (of {len(counts)} seen)")

    feats = os.path.join(args.dstc_dir, "feats")
    streams = ["i3d_rgb", "i3d_flow"]

    def encode_words(text, length):
        ids = [word_to_id.get(w, 0) for w in tokenize(text)][:length]
        return ids + [0] * (length - len(ids))

    with h5py.File(args.output, "w") as h5:
        h5.create_dataset("vocab", data=np.array(words, dtype=h5py.string_dtype()))

        for split, dialogs in splits.items():
            rows, videos, row_of = [], [], {}
            missing = 0
            for dialog in dialogs:
                video = dialog["image_id"]
                paths = [os.path.join(feats, s, f"{video}.npy") for s in streams]
                audio_path = os.path.join(feats, "vggish", f"{video}.npy")
                if not all(os.path.exists(p) for p in paths + [audio_path]):
                    missing += 1
                    continue

                if video not in row_of:
                    row_of[video] = len(videos)
                    videos.append(
                        (
                            np.stack(
                                [fixed(np.load(p).astype(np.float32), args.video_segments)[0] for p in paths]
                            ),
                            fixed(np.load(audio_path).astype(np.float32), args.audio_segments)[0],
                        )
                    )

                history = []
                for index, turn in enumerate(dialog["dialog"]):
                    rows.append(
                        (
                            row_of[video],
                            encode_words(turn["question"], args.max_words),
                            encode_words(turn["answer"], args.max_words),
                            # Most recent turns first, so truncation drops the oldest.
                            encode_words(" ".join(reversed(history)), args.max_history),
                            index,
                            video,
                        )
                    )
                    history.append(f"{turn['question']} {turn['answer']}")

            if missing:
                print(f"  {split}: {missing} dialogs skipped for absent features")
            feature_row, question, answer, history_w, turn_index, ids = zip(*rows)
            video_f, audio_f = zip(*videos)
            h5.create_dataset(f"{split}_video", data=np.stack(video_f), dtype=np.float32)
            h5.create_dataset(f"{split}_audio", data=np.stack(audio_f), dtype=np.float32)
            h5.create_dataset(f"{split}_feature_row", data=np.asarray(feature_row, dtype=np.int32))
            h5.create_dataset(f"{split}_question", data=np.asarray(question, dtype=np.int32))
            h5.create_dataset(f"{split}_answer", data=np.asarray(answer, dtype=np.int32))
            h5.create_dataset(f"{split}_history", data=np.asarray(history_w, dtype=np.int32))
            h5.create_dataset(f"{split}_turn", data=np.asarray(turn_index, dtype=np.int32))
            h5.create_dataset(f"{split}_video_id", data=np.array(ids, dtype=h5py.string_dtype()))
            print(f"{split}: {len(rows)} turns over {len(videos)} videos")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
