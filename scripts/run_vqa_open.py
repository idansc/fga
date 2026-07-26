#!/usr/bin/env python
"""Train and evaluate open-ended VQA v1.

Reuses the files `scripts/prepare_vqa.py` builds — VQA v1's open-ended and
multiple-choice tracks share the same questions and annotations, so the only
differences from the MC run are that the candidates are not fed to the model and
prediction is an unrestricted argmax over the whole answer vocabulary.

```bash
python scripts/run_vqa_open.py --vqa_dir vqa --output_dir models/vqa-open \
    --do_train --do_eval --per_device_train_batch_size 128 \
    --learning_rate 7e-4 --num_train_epochs 8 --bf16
```
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.vqa import OpenEndedVQAConfig, OpenEndedVQAModel  # noqa: E402
from fga.tasks.vqa.data import VQACollator, VQAMultipleChoiceDataset, load_vocab, vqa_accuracy  # noqa: E402

logger = logging.getLogger(__name__)


class OpenEndedView(Dataset):
    """The MC dataset without the candidates, which open-ended never sees."""

    def __init__(self, base: VQAMultipleChoiceDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        item.pop("choice_input_ids")
        return item


@dataclass
class Arguments:
    vqa_dir: str = field(default="vqa")
    hidden_size: int = field(default=512)
    pooling_dim: int = field(default=16000)
    features_in_memory: bool = field(default=True)
    max_eval_questions: Optional[int] = field(default=None)


def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    set_seed(training_args.seed)

    vocab = load_vocab(os.path.join(args.vqa_dir, "vocab.json"))
    with open(os.path.join(args.vqa_dir, "val_human_answers.json")) as f:
        human_answers = {int(k): v for k, v in json.load(f).items()}

    def make(split, in_memory):
        return VQAMultipleChoiceDataset(
            vqa_h5_path=os.path.join(args.vqa_dir, "vqa_mc.h5"),
            features_h5_path=os.path.join(args.vqa_dir, "features.h5"),
            split=split,
            in_memory=in_memory,
        )

    train_base = make("train", args.features_in_memory) if training_args.do_train else None
    eval_base = make("val", False)
    if args.max_eval_questions:
        for attr in ("questions", "choices", "labels", "feature_rows", "question_ids"):
            setattr(eval_base, attr, getattr(eval_base, attr)[: args.max_eval_questions])

    sample = eval_base[0]
    num_regions, feature_dim = sample["image_features"].shape

    model = OpenEndedVQAModel(
        OpenEndedVQAConfig(
            vocab_size=len(vocab["words"]) + 1,
            num_answers=len(vocab["answers"]) + 1,
            hidden_size=args.hidden_size,
            word_embed_dim=args.hidden_size,
            image_feature_dim=feature_dim,
            num_regions=num_regions,
            max_question_length=eval_base.questions.shape[1],
            pooling_dim=args.pooling_dim,
        )
    )
    logger.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    answers = vocab["answers"]
    question_ids = eval_base.question_ids

    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = torch.as_tensor(np.asarray(logits, dtype=np.float32))
        labels = torch.as_tensor(np.asarray(eval_prediction.label_ids))

        # Unrestricted argmax over the vocabulary; id 0 (pad/OOV) is not an answer.
        logits[:, 0] = float("-inf")
        picked = logits.argmax(dim=-1)

        scored = labels > 0
        exact = (picked[scored] == labels[scored]).float().mean().item()
        official = [
            vqa_accuracy(answers[p - 1], human_answers[int(qid)])
            for p, qid in zip(picked.tolist(), question_ids[: logits.size(0)].tolist())
        ]
        return {"exact_accuracy": exact, "vqa_accuracy": float(np.mean(official))}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=OpenEndedView(train_base) if train_base else None,
        eval_dataset=OpenEndedView(eval_base),
        data_collator=VQACollator(),
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
