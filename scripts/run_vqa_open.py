#!/usr/bin/env python
"""Train and evaluate open-ended VQA v1.

Reuses the files `scripts/prepare_vqa.py` builds — VQA v1's open-ended and
multiple-choice tracks share the same questions and annotations, so the only
differences from the MC run are that the candidates are not fed to the model and
prediction is an unrestricted argmax over the whole answer vocabulary.

```bash
python scripts/run_vqa_open.py --vqa_dir vqa --output_dir models/vqa-open \
    --do_train --do_eval --per_device_train_batch_size 512 \
    --learning_rate 2e-3 --num_train_epochs 20 --bf16
```

The published numbers for this task are measured on **test-dev**, whose labels are
not public, after training on train2014 *and* val2014. Training here is on
train2014 alone and scoring is on val2014, which is the only protocol that can be
run without submitting to the evaluation server -- so the numbers are not directly
comparable, and the difference is not only the evaluation set but about 50% more
training data.

`--val_holdout` reserves that fraction of the val *images* for scoring, and
`--train_on_val` adds the rest to training. Running the pair measures how much of
the difference is the data: both score exactly the same held-out questions, and
only one has seen the other 80% of val. Splitting by image, not by question, keeps
every question about a given picture on one side.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
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
    loss_type: str = field(default="soft_ce", metadata={"help": "soft_ce | bce | ce"})
    gated_tanh: bool = field(default=True)
    mask_padding: bool = field(default=True)
    dropout: float = field(default=0.5)
    classifier_dropout: float = field(default=0.3)
    features_in_memory: bool = field(default=True)
    normalize_features: bool = field(default=True)
    max_eval_questions: Optional[int] = field(default=None)
    val_holdout: float = field(
        default=0.0,
        metadata={"help": "Fraction of val *images* to reserve for scoring. 0 scores all of val."},
    )
    train_on_val: bool = field(
        default=False,
        metadata={"help": "Add the val images not held out to training, the paper's train+val protocol."},
    )


def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    set_seed(training_args.seed)

    vocab = load_vocab(os.path.join(args.vqa_dir, "vocab.json"))
    with open(os.path.join(args.vqa_dir, "val_human_answers.json")) as f:
        human_answers = {int(k): v for k, v in json.load(f).items()}

    num_answers = len(vocab["answers"]) + 1  # 0 is pad/OOV

    def make(split, in_memory):
        return VQAMultipleChoiceDataset(
            vqa_h5_path=os.path.join(args.vqa_dir, "vqa_mc.h5"),
            features_h5_path=os.path.join(args.vqa_dir, "features.h5"),
            split=split,
            in_memory=in_memory,
            # Graded objectives need the per-answer scores; "ce" does not.
            num_answers=num_answers if args.loss_type != "ce" else None,
            normalize_features=args.normalize_features,
        )

    train_base = make("train", args.features_in_memory) if training_args.do_train else None
    eval_base = make("val", False)
    if args.max_eval_questions:
        for attr in ("questions", "choices", "labels", "feature_rows", "question_ids"):
            setattr(eval_base, attr, getattr(eval_base, attr)[: args.max_eval_questions])

    # Split val by image so that the same held-out questions can be scored by a
    # model trained with the rest of val and by one trained without it.
    val_train_indices, val_eval_indices = None, None
    if args.val_holdout > 0:
        # The split depends only on the seed, so a run with train_on_val and one
        # without score exactly the same questions.
        images = np.unique(eval_base.question_ids // 10)
        rng = np.random.default_rng(training_args.seed)
        rng.shuffle(images)
        held_out = set(images[: int(len(images) * args.val_holdout)].tolist())
        is_held = np.fromiter(((q // 10) in held_out for q in eval_base.question_ids), bool, len(eval_base))
        val_eval_indices = np.flatnonzero(is_held)
        val_train_indices = np.flatnonzero(~is_held) if args.train_on_val else None
        logger.info(
            f"val split by image: {len(val_eval_indices)} questions held out over {len(held_out)} images"
            + (f"; {len(val_train_indices)} join training" if val_train_indices is not None else "")
        )

    sample = eval_base[0]
    num_regions, feature_dim = sample["image_features"].shape

    model = OpenEndedVQAModel(
        OpenEndedVQAConfig(
            vocab_size=len(vocab["words"]) + 1,
            num_answers=num_answers,
            hidden_size=args.hidden_size,
            word_embed_dim=args.hidden_size,
            image_feature_dim=feature_dim,
            num_regions=num_regions,
            max_question_length=eval_base.questions.shape[1],
            pooling_dim=args.pooling_dim,
            loss_type=args.loss_type,
            gated_tanh=args.gated_tanh,
            mask_padding=args.mask_padding,
            dropout=args.dropout,
            classifier_dropout=args.classifier_dropout,
        )
    )
    logger.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    answers = vocab["answers"]
    question_ids = eval_base.question_ids if val_eval_indices is None else eval_base.question_ids[val_eval_indices]

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

    # `answer_scores` is in the model's forward signature, so the Trainer would
    # keep it; this is explicit anyway because the equivalent column in the
    # Visual Dialog dense path was silently dropped once and the loss trained on
    # nothing while still looking plausible. The model also raises if the scores
    # go missing during training, which is the check that actually catches it.
    training_args.remove_unused_columns = False

    train_dataset = OpenEndedView(train_base) if train_base else None
    eval_dataset = OpenEndedView(eval_base)
    if val_eval_indices is not None:
        if train_dataset is not None and val_train_indices is not None:
            train_dataset = ConcatDataset(
                [train_dataset, Subset(OpenEndedView(eval_base), val_train_indices.tolist())]
            )
        eval_dataset = Subset(eval_dataset, val_eval_indices.tolist())

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
