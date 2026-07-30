#!/usr/bin/env python
"""Prepare DSTC7 Audio-Visual Scene-Aware Dialog.

Turns the challenge's dialog JSONs and per-video feature files into one h5 the
dataset reads. Each of the ten turns in a dialog becomes an example: the question,
the turns before it as history, and the video's features.

`--video_source` decides what the video modalities are, and it is the whole
question of what attention can be asked.

* `spatial` is the paper's own setup: four VGG19 conv grids, 49 regions each,
  from `scripts/extract_spatial_frames.py`. The entities are *regions*, so
  attention answers "where in this frame". This is what `run.sh` in the original
  release calls `i3d_rgb_vgg19_4`, and it drops straight into the four video
  utilities the encoder already has.
* `i3d` uses what DSTC7 distributes. Those are pooled over space -- `i3d_rgb` is
  `(segments, 2048)`, not `(segments, regions, 2048)` -- so the entities are
  *moments*, and the question becomes "when in this video". Different question,
  and not the one the paper asked.
* `both` keeps the pooled streams as the video utilities and adds the frames as
  one extra weight-sharing modality, the way dialog rounds share weights. That is
  past the paper, not a reproduction of it: it asks where and when at once.

Frames dominate the file, so `--num_frames` is what decides its size: at four
frames the table is under two gigabytes, at forty-eight it is twenty-three.

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
import sys
from collections import Counter

import h5py
import numpy as np


def tokenize(text):
    """Split on whitespace, as the original release does.

    The DSTC7 text arrives already lowercased with punctuation spaced out
    ("there is only one person in the video ."), so whitespace is the tokenizer
    the corpus was written for. A regex that keeps only word characters looks
    tidier and quietly drops every full stop and question mark -- which changes
    the vocabulary, and leaves a model that cannot produce the punctuation its
    references contain.
    """
    return text.lower().split()


def fixed(array, count):
    """Resample evenly to exactly `count` rows, covering the whole sequence.

    Truncating to the first `count` rows is the obvious reading of "fixed
    length" and it is wrong here. A VGGish clip runs a median of 62 rows and an
    I3D one 184, so keeping the head means the audio is the opening five seconds
    of a thirty-second video and the video is its first quarter. The questions
    are about the whole clip -- "does she leave at the end?" -- so a snippet of
    the beginning is not weak evidence, it is confident evidence about the wrong
    moment, which is how a modality ends up scoring worse than its own absence.

    Short clips are resampled rather than zero-padded, which is the same argument
    from the other side: a padded row is a zero vector that attention still
    competes over, so padding a twelve-row clip out to sixty-four hands most of
    the distribution to slots describing nothing. Repeating rows evenly keeps
    every slot meaningful. The original sidesteps this by padding only to each
    batch's own maximum and never truncating -- an option a fixed-shape h5 does
    not have.

    The second return value is the number of real rows behind the resampling, for
    callers that want to know how much was repeated.
    """
    take = np.linspace(0, len(array) - 1, count).astype(int)
    return array[take].astype(np.float32), min(len(array), count)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dstc_dir", required=True, help="Holds the dialog JSONs and feats/.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video_segments", type=int, default=48)
    parser.add_argument("--audio_segments", type=int, default=48)
    parser.add_argument("--max_words", type=int, default=20, help="Question tokens; the paper keeps 10.")
    parser.add_argument(
        "--max_answer_words",
        type=int,
        help="Answer tokens. Defaults to --max_words, but the two are worth separating: the question "
        "length is an attention entity count, while truncating an answer just deletes the reference.",
    )
    parser.add_argument("--max_history", type=int, default=10, help="Turns of history kept, most recent last.")
    parser.add_argument("--max_turn_words", type=int, default=20, help="Words kept from each history turn.")
    parser.add_argument("--min_count", type=int, default=2, help="Word frequency floor, from train only.")
    parser.add_argument("--spatial_dir", help="Per-frame conv grids from extract_spatial_frames.py.")
    parser.add_argument("--num_frames", type=int, default=4, help="Frames kept, strided from what was extracted.")
    parser.add_argument(
        "--video_source",
        choices=("spatial", "i3d", "both"),
        default="spatial",
        help="What the video utilities are: conv grids (the paper), pooled DSTC7 streams, or both.",
    )
    args = parser.parse_args()
    if args.video_source in ("spatial", "both") and not args.spatial_dir:
        parser.error(f"--video_source {args.video_source} needs --spatial_dir.")
    max_answer_words = args.max_answer_words or args.max_words

    splits = {}
    for name, filename in (
        ("train", "train_set4DSTC7-AVSD.json"),
        ("val", "valid_set4DSTC7-AVSD.json"),
        ("test", "test_set4DSTC7-AVSD.json"),
    ):
        path = os.path.join(args.dstc_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            splits[name] = json.load(handle)["dialogs"]
        print(f"{name}: {len(splits[name])} dialogs")

    # The test set is shaped differently, and the difference is the point of the
    # benchmark: each dialog is cut at one turn whose answer reads
    # `__UNDISCLOSED__`, with everything before it as history. The cut lands at a
    # different depth per dialog -- 380 at the first turn with no history at all,
    # 95 at the tenth -- so a model is asked to answer from nothing and from nine
    # turns of context in the same evaluation.
    UNDISCLOSED = "__UNDISCLOSED__"
    EOS, UNK = "<eos>", "<unk>"

    # Vocabulary from the training questions and answers only. The captions and
    # summaries are not part of the model's input, and counting them admits words
    # the model can never be asked about while still paying for their embeddings
    # -- the original gates them behind `--include-caption`, which its own run
    # script does not pass.
    counts = Counter()
    for dialog in splits["train"]:
        for turn in dialog["dialog"]:
            counts.update(tokenize(turn["question"]))
            counts.update(tokenize(turn["answer"]))
    # Id 0 is padding, 1 is end-of-answer, 2 is the unknown word.
    #
    # The decoder needs a token it can emit to stop, and padding cannot serve:
    # the loss ignores id 0, so the model is never taught that an answer ended.
    # Without `<eos>` every answer runs to the length cap, which costs nothing in
    # perplexity -- the surplus tokens are never scored -- and wrecks BLEU and
    # CIDEr, which compare lengths directly.
    #
    # `<unk>` has to be its own id for the same reason in reverse. Folding rare
    # words into the padding id makes them vanish: a masked position contributes
    # nothing to an encoder and nothing to the loss, so the sentence silently
    # loses a word rather than recording that an unfamiliar one was there.
    words = [EOS, UNK] + [w for w, c in counts.most_common() if c >= args.min_count]
    word_to_id = {w: i + 1 for i, w in enumerate(words)}
    unk_id = word_to_id[UNK]
    print(f"{len(words) - 2} words (of {len(counts)} seen), plus {EOS} and {UNK}")

    feats = os.path.join(args.dstc_dir, "feats")
    streams = ["i3d_rgb", "i3d_flow"]

    def encode_words(text, length):
        ids = [word_to_id.get(w, unk_id) for w in tokenize(text)][:length]
        return ids + [0] * (length - len(ids))

    def encode_answer(text, length):
        """Like `encode_words`, but ending in `<eos>` so the decoder learns to stop."""
        ids = [word_to_id.get(w, unk_id) for w in tokenize(text)][: length - 1]
        ids.append(word_to_id[EOS])
        return ids + [0] * (length - len(ids))

    with h5py.File(args.output, "w") as h5:
        h5.create_dataset("vocab", data=np.array(words, dtype=h5py.string_dtype()))

        for split, dialogs in splits.items():
            rows, videos, frames, row_of = [], [], [], {}
            missing = 0
            # Test features ship in their own directories; a few test videos are
            # absent from the train/val ones, so both are searched.
            suffixes = ("_testset", "") if split == "test" else ("",)

            def find(kind, video):
                """The first existing path for this feature kind, or the plain one."""
                candidates = [os.path.join(feats, kind + s, f"{video}.npy") for s in suffixes]
                return next((p for p in candidates if os.path.exists(p)), candidates[-1])

            for dialog in dialogs:
                video = dialog["image_id"]
                paths = [find(s, video) for s in streams]
                audio_path = find("vggish", video)
                spatial_path = os.path.join(args.spatial_dir, f"{video}.npy") if args.spatial_dir else None
                needed = [audio_path]
                if args.video_source in ("i3d", "both"):
                    needed += paths
                if spatial_path:
                    needed.append(spatial_path)
                if not all(os.path.exists(p) for p in needed):
                    missing += 1
                    continue

                if video not in row_of:
                    row_of[video] = len(videos)
                    if spatial_path:
                        # The four frames the paper uses are strided across
                        # whatever the extractor wrote, so one extraction serves
                        # any frame count.
                        grid = np.load(spatial_path)
                        take = np.linspace(0, len(grid) - 1, args.num_frames).astype(int)
                        sampled = grid[take]

                    if args.video_source == "spatial":
                        video_streams = sampled.astype(np.float32)
                    else:
                        video_streams = np.stack(
                            [fixed(np.load(p).astype(np.float32), args.video_segments)[0] for p in paths]
                        )
                    videos.append(
                        (video_streams, fixed(np.load(audio_path).astype(np.float32), args.audio_segments)[0])
                    )
                    if args.video_source == "both":
                        # Kept float16, as extracted: casting here would double a
                        # table that is already the largest thing in the file.
                        frames.append(sampled)

                history = []
                for index, turn in enumerate(dialog["dialog"]):
                    undisclosed = turn["answer"] == UNDISCLOSED
                    # Every turn is an example while the answer is known; on test
                    # only the withheld one is, since the rest are given as
                    # context, not as questions.
                    if split != "test" or undisclosed:
                        rows.append(
                            (
                                row_of[video],
                                encode_words(turn["question"], args.max_words),
                                # A withheld answer becomes all padding: the loss
                                # ignores id 0, so it contributes nothing, and
                                # generation never reads it.
                                [0] * max_answer_words if undisclosed else encode_answer(turn["answer"], max_answer_words),
                                # Turns, not a flattened word sequence: the
                                # history encoder reads each question-answer pair
                                # on its own and then reads the pairs in order.
                                # Padding turns lead, so the real ones end at the
                                # most recent -- which is the state the pair-level
                                # LSTM finishes on.
                                ([[0] * args.max_turn_words] * (args.max_history - len(history))
                                 + history[-args.max_history:]),
                                index,
                                video,
                            )
                        )
                    history.append(encode_words(f"{turn['question']} {turn['answer']}", args.max_turn_words))

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
            if frames:
                # Written one video at a time: stacking first would need a second
                # copy of a table measured in gigabytes.
                table = h5.create_dataset(
                    f"{split}_frames",
                    shape=(len(frames), *frames[0].shape),
                    dtype=np.float16,
                    chunks=(1, *frames[0].shape),
                )
                for row, grid in enumerate(frames):
                    table[row] = grid
                print(f"{split}: frames {table.shape}")
            print(f"{split}: {len(rows)} turns over {len(videos)} videos")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
