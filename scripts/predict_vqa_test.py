#!/usr/bin/env python
"""Write a VQA test2015 submission file from a trained open-ended model.

VQA v1's test answers were never released, so a model trained on train+val --
the protocol behind the published test-dev numbers -- cannot be scored locally.
What it can do is emit the file the evaluation server takes, which reports both
test-dev and test-standard from one submission over the full test2015 set.

The output is the format the server expects:

    [{"question_id": 4195880, "answer": "yes"}, ...]

```bash
python scripts/predict_vqa_test.py \
    --model models/open-trainval --vqa_dir vqa \
    --questions vqa/raw/OpenEnded_mscoco_test2015_questions.json \
    --features vqa/test_features.h5 --output vqa/test2015_results.json
```
"""

import argparse
import json
import os
import re
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.vqa import OpenEndedVQAModel  # noqa: E402
from fga.tasks.vqa.data import load_vocab  # noqa: E402

TOKEN = re.compile(r"[a-z0-9']+")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vqa_dir", required=True, help="Holds vocab.json, so tokenization matches training.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--features", required=True, help="h5 with `features` and `coco_ids`, from prepare_vqa.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    vocab = load_vocab(os.path.join(args.vqa_dir, "vocab.json"))
    word_to_id = {w: i + 1 for i, w in enumerate(vocab["words"])}
    answers = vocab["answers"]

    model = OpenEndedVQAModel.from_pretrained(args.model).to(args.device).eval()
    length = model.config.max_question_length

    with open(args.questions) as handle:
        questions = json.load(handle)["questions"]
    print(f"{len(questions)} questions over {len({q['image_id'] for q in questions})} images")

    with h5py.File(args.features, "r") as h5:
        row_of = {int(c): r for r, c in enumerate(h5["coco_ids"][:])}
        missing = {q["image_id"] for q in questions} - set(row_of)
        if missing:
            raise SystemExit(f"{len(missing)} test images have no features, e.g. {sorted(missing)[:3]}")

        results = []
        for start in range(0, len(questions), args.batch_size):
            batch = questions[start : start + args.batch_size]

            token_ids = np.zeros((len(batch), length), dtype=np.int64)
            for i, question in enumerate(batch):
                ids = [word_to_id.get(w, 0) for w in TOKEN.findall(question["question"].lower())][:length]
                token_ids[i, : len(ids)] = ids

            # h5py fancy indexing needs strictly increasing indices, and several
            # questions share an image -- so gather each row once, then expand.
            rows = np.asarray([row_of[q["image_id"]] for q in batch])
            unique, inverse = np.unique(rows, return_inverse=True)
            features = h5["features"][unique.tolist()].astype(np.float32)[inverse]
            # The same per-region normalization the dataset applies in training.
            features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-6)

            with torch.no_grad():
                logits = model(
                    question_input_ids=torch.from_numpy(token_ids).to(args.device),
                    image_features=torch.from_numpy(features).to(args.device),
                ).logits.float()
            logits[:, 0] = float("-inf")  # id 0 is padding, not an answer
            picked = logits.argmax(dim=-1).cpu().numpy()

            results.extend(
                {"question_id": int(q["question_id"]), "answer": answers[p - 1]}
                for q, p in zip(batch, picked)
            )
            if start % (args.batch_size * 100) == 0:
                print(f"  {start}/{len(questions)}", flush=True)

    with open(args.output, "w") as handle:
        json.dump(results, handle)
    print(f"wrote {args.output}: {len(results)} answers")
    print("Upload to the VQA evaluation server; it reports test-dev and test-standard.")


if __name__ == "__main__":
    sys.exit(main())
