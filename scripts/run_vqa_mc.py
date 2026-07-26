#!/usr/bin/env python
"""Train and evaluate multiple-choice VQA v1 with high-order attention.

```bash
python scripts/run_vqa_mc.py \
    --vqa_dir vqa --output_dir models/vqa-mc \
    --do_train --do_eval --per_device_train_batch_size 128 \
    --learning_rate 7e-4 --num_train_epochs 8 --bf16
```

Evaluation reports two numbers. `mc_accuracy` is exact match of the predicted
answer id against the annotators' consensus answer, over the questions whose
consensus answer is in the vocabulary. `vqa_accuracy` is the official graded
metric, computed by restricting the argmax to each question's 18 candidates and
scoring the chosen string against the ten human answers — averaged over the
leave-one-annotator-out subsets, with the official answer normalization, and
counting a question with no in-vocabulary candidate as wrong rather than
excusing it.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.vqa import HighOrderAttentionConfig, HighOrderAttentionForVQA  # noqa: E402
from fga.tasks.vqa.data import VQACollator, VQAMultipleChoiceDataset, load_vocab, vqa_accuracy  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    vqa_dir: str = field(default="vqa", metadata={"help": "Output of scripts/prepare_vqa.py."})
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Resume from a trained model."})
    hidden_size: int = field(default=512)
    pooling_dim: int = field(default=16000)
    use_ternary: bool = field(default=True)
    mask_padding: bool = field(default=True, metadata={"help": "Keep attention off padded words and slots."})
    features_in_memory: bool = field(default=True, metadata={"help": "~18 GB as float16."})
    normalize_features: bool = field(default=True, metadata={"help": "L2-normalize each region."})
    max_eval_questions: Optional[int] = field(default=None)


def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    set_seed(training_args.seed)

    vocab = load_vocab(os.path.join(args.vqa_dir, "vocab.json"))
    with open(os.path.join(args.vqa_dir, "val_human_answers.json")) as f:
        human_answers = {int(k): v for k, v in json.load(f).items()}

    def make(split):
        return VQAMultipleChoiceDataset(
            vqa_h5_path=os.path.join(args.vqa_dir, "vqa_mc.h5"),
            features_h5_path=os.path.join(args.vqa_dir, "features.h5"),
            split=split,
            in_memory=args.features_in_memory and split == "train",
            normalize_features=args.normalize_features,
        )

    train_dataset = make("train") if training_args.do_train else None
    eval_dataset = make("val")
    if args.max_eval_questions:
        eval_dataset.questions = eval_dataset.questions[: args.max_eval_questions]
        eval_dataset.choices = eval_dataset.choices[: args.max_eval_questions]
        eval_dataset.labels = eval_dataset.labels[: args.max_eval_questions]
        eval_dataset.feature_rows = eval_dataset.feature_rows[: args.max_eval_questions]
        eval_dataset.question_ids = eval_dataset.question_ids[: args.max_eval_questions]

    sample = eval_dataset[0]
    num_regions, feature_dim = sample["image_features"].shape

    if args.model_name_or_path:
        model = HighOrderAttentionForVQA.from_pretrained(args.model_name_or_path)
    else:
        model = HighOrderAttentionForVQA(
            HighOrderAttentionConfig(
                vocab_size=len(vocab["words"]) + 1,
                num_answers=len(vocab["answers"]) + 1,  # 0 is pad/OOV
                hidden_size=args.hidden_size,
                word_embed_dim=args.hidden_size,
                image_feature_dim=feature_dim,
                num_regions=num_regions,
                max_question_length=eval_dataset.questions.shape[1],
                num_choices=eval_dataset.choices.shape[1],
                pooling_dim=args.pooling_dim,
                use_ternary=args.use_ternary,
                mask_padding=args.mask_padding,
            )
        )
    logger.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    answers = vocab["answers"]
    choices_table = eval_dataset.choices
    question_ids = eval_dataset.question_ids

    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = torch.as_tensor(np.asarray(logits, dtype=np.float32))
        labels = torch.as_tensor(np.asarray(eval_prediction.label_ids))

        # Restrict the argmax to each question's candidates; padding/OOV slots
        # (id 0) are masked out, so an out-of-vocabulary choice cannot be picked.
        choices = torch.as_tensor(choices_table[: logits.size(0)].astype(np.int64))
        candidate_scores = logits.gather(1, choices.clamp(min=0))
        candidate_scores[choices == 0] = float("-inf")
        picked = choices.gather(1, candidate_scores.argmax(dim=1, keepdim=True)).squeeze(1)

        scored = labels > 0
        mc_accuracy = (picked[scored] == labels[scored]).float().mean().item()

        # A question whose eighteen candidates are all out of vocabulary cannot be
        # answered at all; it scores zero rather than being dropped from the mean.
        official = [
            vqa_accuracy(answers[p - 1], human_answers[int(qid)]) if p > 0 else 0.0
            for p, qid in zip(picked.tolist(), question_ids[: logits.size(0)].tolist())
        ]
        return {"mc_accuracy": mc_accuracy, "vqa_accuracy": float(np.mean(official))}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
