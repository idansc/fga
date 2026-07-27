#!/usr/bin/env python
"""Prepare multiple-choice VQA v1 for training.

Builds, from the official release and a directory of per-image feature files:

    features.h5   (num_images, num_regions, dim) float16, plus the COCO ids
    vqa_mc.h5     tokenized questions, choice ids, labels, feature row per
                  question, and the soft answer scores
    vocab.json    question words and the answer vocabulary

Answers: the top `--num_answers` multiple-choice answers of the train split, ids
`1..N` with 0 reserved for padding/out-of-vocabulary. Train questions whose
correct answer is out of vocabulary are dropped (they could only teach the wrong
thing); val keeps every question, and also stores the ten human answers so the
official VQA accuracy can be computed.

```bash
python scripts/prepare_vqa.py --raw_dir vqa/raw \
    --features_dir data/extracted/train --output_dir vqa
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.vqa.data import normalize_answer, vqa_accuracy  # noqa: E402

TOKEN = re.compile(r"[a-z0-9']+")

#: No VQA question has more than ten annotators, so ten distinct answers is the cap.
MAX_TARGETS = 10


def tokenize(text: str):
    return TOKEN.findall(text.lower())


def build_feature_table(features_path: str, needed: list, args) -> dict:
    """Write one feature row per referenced COCO image, and return id -> row.

    The table is tens of gigabytes and takes far longer to build than everything
    else here, so an existing one covering exactly the images this run needs is
    reused, and `--features_dir` is not needed in that case. It is checked against
    `needed` rather than trusted: a table built for a different question set would
    pair every question with the wrong picture and nothing downstream would
    notice.
    """
    if os.path.exists(features_path) and not args.rebuild_features:
        with h5py.File(features_path, "r") as h5:
            existing = [int(c) for c in h5["coco_ids"][:]]
        if existing != needed:
            raise SystemExit(
                f"{features_path} covers {len(existing)} images but this run needs {len(needed)}, "
                "or covers different ones. Pass --rebuild_features."
            )
        print(f"reusing {features_path}: {len(existing)} images")
        return {coco_id: row for row, coco_id in enumerate(existing)}

    if not args.features_dir:
        raise SystemExit(f"{features_path} does not exist; pass --features_dir to build it.")

    stem_of = {}
    for name in os.listdir(args.features_dir):
        if name.endswith(".npy") and not name.endswith("_info.npy"):
            match = re.search(r"(\d+)\.npy$", name)
            if match:
                stem_of[int(match.group(1))] = os.path.join(args.features_dir, name)
    missing = [i for i in needed if i not in stem_of]
    if missing:
        raise SystemExit(f"{len(missing)} images have no features, e.g. {missing[:5]}")

    row_of = {}
    with h5py.File(features_path, "w") as h5:
        sample = np.load(stem_of[needed[0]])
        table = h5.create_dataset("features", (len(needed), args.num_regions, sample.shape[1]), dtype=np.float16)
        h5.create_dataset("coco_ids", data=np.asarray(needed, dtype=np.int64))
        for row, coco_id in enumerate(needed):
            feats = np.load(stem_of[coco_id])[: args.num_regions]
            table[row, : len(feats)] = feats.astype(np.float16)
            row_of[coco_id] = row
            if row % 20000 == 0:
                print(f"  features {row}/{len(needed)}", flush=True)
    print(f"wrote {features_path}: {len(needed)} images")
    return row_of


def soft_targets(human_answers, answer_to_id: dict):
    """Every answer an annotator gave, with the VQA score it would earn.

    Returns `(ids, scores)` padded to [`MAX_TARGETS`]. Answers outside the
    vocabulary are dropped -- the model cannot name them, so training it toward
    them only teaches it to spread mass over answers it will never emit.
    """
    normalized = [normalize_answer(a) for a in human_answers]
    ids, scores = [], []
    for answer in dict.fromkeys(normalized):  # distinct, order preserved
        answer_id = answer_to_id.get(answer)
        if answer_id is None:
            continue
        ids.append(answer_id)
        scores.append(vqa_accuracy(answer, normalized, normalize=False))
    ids = ids[:MAX_TARGETS]
    scores = scores[:MAX_TARGETS]
    return (
        ids + [0] * (MAX_TARGETS - len(ids)),
        scores + [0.0] * (MAX_TARGETS - len(scores)),
    )


def load_pairs(raw_dir: str, split: str):
    with open(os.path.join(raw_dir, f"MultipleChoice_mscoco_{split}_questions.json")) as f:
        questions = json.load(f)["questions"]
    with open(os.path.join(raw_dir, f"mscoco_{split}_annotations.json")) as f:
        annotations = {a["question_id"]: a for a in json.load(f)["annotations"]}
    return questions, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument(
        "--features_dir",
        default=None,
        help="Per-image .npy files named after the COCO images. Only needed when features.h5 must be built.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_answers", type=int, default=3000)
    parser.add_argument("--max_question_length", type=int, default=15)
    parser.add_argument("--num_choices", type=int, default=18)
    parser.add_argument("--num_regions", type=int, default=36)
    parser.add_argument("--rebuild_features", action="store_true", help="Ignore an existing features.h5.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_q, train_a = load_pairs(args.raw_dir, "train2014")
    val_q, val_a = load_pairs(args.raw_dir, "val2014")

    # --- vocabularies, from train only ---
    answer_counts = Counter(a["multiple_choice_answer"] for a in train_a.values())
    answers = [a for a, _ in answer_counts.most_common(args.num_answers)]
    answer_to_id = {a: i + 1 for i, a in enumerate(answers)}  # 0 = pad / OOV

    word_counts = Counter(w for q in train_q for w in tokenize(q["question"]))
    words = [w for w, c in word_counts.most_common() if c >= 2]
    word_to_id = {w: i + 1 for i, w in enumerate(words)}  # 0 = pad; rare words drop to 0

    with open(os.path.join(args.output_dir, "vocab.json"), "w") as f:
        json.dump({"words": words, "answers": answers}, f)
    covered = sum(answer_counts[a] for a in answers) / sum(answer_counts.values())
    print(f"{len(words)} question words; {len(answers)} answers cover {covered:.1%} of train")

    # The vocabulary is keyed on the raw answer strings, so that the
    # multiple-choice candidates -- also raw -- look up directly. Soft targets
    # come from the annotators' free-form answers, which have to go through the
    # official normalization first, so they need a second map keyed that way.
    # Where two vocabulary entries normalize alike the more frequent one wins,
    # since `answers` is already in descending frequency order.
    normalized_to_id = {}
    for answer, answer_id in answer_to_id.items():
        normalized_to_id.setdefault(normalize_answer(answer), answer_id)

    # --- feature table, one row per referenced COCO image ---
    needed = sorted({q["image_id"] for q in train_q} | {q["image_id"] for q in val_q})
    features_path = os.path.join(args.output_dir, "features.h5")
    row_of = build_feature_table(features_path, needed, args)

    # --- encode both splits ---
    def encode(questions, annotations, split: str, drop_oov_labels: bool):
        rows = []
        for q in questions:
            ann = annotations[q["question_id"]]
            label = answer_to_id.get(ann["multiple_choice_answer"], 0)
            if drop_oov_labels and label == 0:
                continue
            ids = [word_to_id.get(w, 0) for w in tokenize(q["question"])][: args.max_question_length]
            ids += [0] * (args.max_question_length - len(ids))
            choices = [answer_to_id.get(c, 0) for c in q["multiple_choices"][: args.num_choices]]
            choices += [0] * (args.num_choices - len(choices))
            target_ids, target_scores = soft_targets([a["answer"] for a in ann["answers"]], normalized_to_id)
            rows.append((q["question_id"], row_of[q["image_id"]], ids, choices, label, target_ids, target_scores))
        return rows

    human = {q["question_id"]: [a["answer"] for a in val_a[q["question_id"]]["answers"]] for q in val_q}
    with open(os.path.join(args.output_dir, "val_human_answers.json"), "w") as f:
        json.dump(human, f)

    with h5py.File(os.path.join(args.output_dir, "vqa_mc.h5"), "w") as h5:
        for split, rows in (
            ("train", encode(train_q, train_a, "train", drop_oov_labels=True)),
            ("val", encode(val_q, val_a, "val", drop_oov_labels=False)),
        ):
            qid, feat_row, ids, choices, labels, target_ids, target_scores = zip(*rows)
            h5.create_dataset(f"{split}_question_ids", data=np.asarray(qid, dtype=np.int64))
            h5.create_dataset(f"{split}_feature_rows", data=np.asarray(feat_row, dtype=np.int64))
            h5.create_dataset(f"{split}_questions", data=np.asarray(ids, dtype=np.int32))
            h5.create_dataset(f"{split}_choices", data=np.asarray(choices, dtype=np.int32))
            h5.create_dataset(f"{split}_labels", data=np.asarray(labels, dtype=np.int64))
            h5.create_dataset(f"{split}_target_ids", data=np.asarray(target_ids, dtype=np.int32))
            h5.create_dataset(f"{split}_target_scores", data=np.asarray(target_scores, dtype=np.float32))
            reachable = np.asarray(target_scores).max(axis=1)
            print(f"{split}: {len(rows)} questions; soft targets reach {100 * (reachable > 0).mean():.1f}%, "
                  f"mean best score {reachable.mean():.3f}")

    print("done")


if __name__ == "__main__":
    sys.exit(main())
