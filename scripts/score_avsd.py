#!/usr/bin/env python
"""Generate DSTC7-AVSD test answers and score them the way the challenge did.

Perplexity is not what AVSD is ranked on. The organisers score a generated answer
against six human references with BLEU, METEOR, ROUGE-L and CIDEr, and a model can
improve its perplexity while writing worse answers, or the reverse.

The official `dstc7avsd_eval.sh` is Python 2 and shells out to a `coco-caption`
checkout. This does the same arithmetic on the same reference file through
`pycocoevalcap`, which is that code packaged for Python 3, and reproduces the two
preprocessing steps that would otherwise quietly change the numbers:

* only the withheld turn of each dialog is scored (`get_hypotheses.py -l`),
* hypotheses pass the organisers' stopword substitutions first (`-s`), which map
  contractions and fillers so that "isn't" and "is not" are not counted as a miss.

Hypotheses are matched to references by name -- `{video}_{turn}` -- rather than by
position. The official script assigns ids by enumeration order, which is the same
thing only as long as the result file happens to be ordered like the reference.

```bash
python scripts/score_avsd.py --model models/fx-spatial-s1 --data avsd/avsd_test.h5 \
    --references avsd/dstc7/dstc7avsd_eval/data/test_set4DSTC7-AVSD_multiref.json \
    --stopwords avsd/dstc7/dstc7avsd_eval/data/stopwords.txt
```
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.video_dialog import AVSDForResponseGeneration  # noqa: E402
from fga.tasks.video_dialog.data import AVSDCollator, AVSDDataset, load_vocab, render  # noqa: E402


class StopwordFilter:
    """The organisers' word substitutions, as `stopword_filter.py` applies them.

    Each line is either a word to delete or a word and its replacement. A word is
    rewritten by the first pattern that changes it, and words rewritten to nothing
    are dropped.
    """

    def __init__(self, path):
        self.patterns = []
        if path and os.path.exists(path):
            for line in open(path):
                fields = line.split()
                if len(fields) == 1:
                    self.patterns.append((re.compile(rf"^{fields[0]}$"), ""))
                elif len(fields) == 2:
                    self.patterns.append((re.compile(rf"^{fields[0]}$"), fields[1]))

    def __call__(self, sentence):
        kept = []
        for word in sentence.split():
            target = word
            for pattern, replacement in self.patterns:
                substituted = pattern.sub(replacement, word)
                if substituted != word:
                    target = substituted
                    break
            if target:
                kept.append(target)
        return " ".join(kept)


def generate(model, dataset, words, device, batch_size=64, beam_width=3, length_penalty=1.0):
    """An answer per row, as `{video}_{turn}: sentence`."""
    collate = AVSDCollator()
    model.eval().to(device)
    answers = {}
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = {k: v.to(device) for k, v in collate(rows).items()}
        inputs = (
            batch["question_input_ids"], batch["history_input_ids"],
            batch["video_features"], batch["audio_features"], batch.get("frame_features"),
        )
        produced = (
            model.beam_search(*inputs, beam_width=beam_width, length_penalty=length_penalty)
            if beam_width else model.generate_answers(*inputs)
        ).cpu().numpy()
        for offset, ids in enumerate(produced):
            index = start + offset
            video = dataset.video_id[index]
            video = video.decode() if isinstance(video, bytes) else video
            answers[f"{video}_{dataset.turn[index]}"] = render(ids, words)
    return answers


def score(answers, reference_path, stopwords_path):
    """BLEU 1-4, METEOR, ROUGE-L and CIDEr against the multi-reference file."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    reference = json.load(open(reference_path))
    id_of = {image["name"]: image["id"] for image in reference["images"]}

    missing = [name for name in id_of if name not in answers]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(id_of)} test questions have no generated answer "
            f"(e.g. {missing[:3]}). Scoring a partial set would silently inflate every metric."
        )

    truth = {image_id: [] for image_id in id_of.values()}
    for annotation in reference["annotations"]:
        truth[annotation["image_id"]].append({"caption": annotation["caption"]})

    swfilter = StopwordFilter(stopwords_path)
    hypotheses = {id_of[name]: [{"caption": swfilter(answers[name])}] for name in id_of}

    tokenizer = PTBTokenizer()
    truth, hypotheses = tokenizer.tokenize(truth), tokenizer.tokenize(hypotheses)

    results = {}
    for scorer, names in (
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE_L"]),
        (Cider(), ["CIDEr"]),
    ):
        value, _ = scorer.compute_score(truth, hypotheses)
        for name, single in zip(names, value if isinstance(value, list) else [value]):
            results[name] = single
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="A directory saved by run_avsd.py.")
    parser.add_argument("--data", required=True, help="An h5 carrying a `test` split.")
    parser.add_argument("--references", required=True, help="test_set4DSTC7-AVSD_multiref.json")
    parser.add_argument("--stopwords", default="", help="The organisers' stopwords.txt.")
    parser.add_argument("--output", help="Where to write the generated answers.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beam_width", type=int, default=3, help="0 for greedy decoding.")
    parser.add_argument("--length_penalty", type=float, default=1.0)
    args = parser.parse_args()

    words = load_vocab(args.data)
    dataset = AVSDDataset(args.data, "test")
    model = AVSDForResponseGeneration.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    answers = generate(model, dataset, words, device, args.batch_size, args.beam_width, args.length_penalty)
    print(f"generated {len(answers)} answers")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(answers, handle, indent=1)

    for name, value in score(answers, args.references, args.stopwords).items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    sys.exit(main())
