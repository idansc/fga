#!/usr/bin/env python
"""Finetune a trained FGA on the dense relevance annotations, to optimize NDCG.

The sparse objective treats one candidate as correct and 99 as equally wrong;
NDCG scores against the graded relevance annotators gave every candidate. This
finetunes on that graded signal over the 2,000-image dense subset of *train*.
The val annotations are never trained on -- they are the evaluation set.

```bash
python scripts/finetune_dense.py \
    --model_name_or_path models/fga-frcnn/checkpoint-48160 \
    --image_features_path data/frcnn_features_new.h5 \
    --train_dense_path data/visdial_1.0_train_dense_annotations.json \
    --output_dir models/fga-ndcg --loss soft_ce --sparse_weight 0.0
```

Expect MRR to fall as NDCG rises: the two metrics genuinely disagree, which is
why challenge entries usually ensemble an MRR-tuned model with an NDCG-tuned one
rather than trying to win both at once.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser, set_seed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga import FGAForVisualDialog, VisDialCollator, VisDialDataset  # noqa: E402
from fga.tasks.visual_dialog.data import (  # noqa: E402
    image_ids_from_params,
    load_visdial_params,
    vocab_size_from_params,
)
from fga.tasks.visual_dialog.dense import DenseFinetuneTrainer, DenseVisDialDataset  # noqa: E402
from fga.tasks.visual_dialog.metrics import build_compute_metrics  # noqa: E402
from fga.tasks.visual_dialog.trainer import FGATrainingArguments, load_dense_annotations  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    model_name_or_path: str = field(metadata={"help": "Trained checkpoint to finetune."})
    image_features_path: str = field(default="data/frcnn_features_new.h5")
    visdial_data_path: str = field(default="data/visdial_data.h5")
    visdial_params_path: str = field(default="data/visdial_params.json")
    train_dense_path: str = field(
        default="data/visdial_1.0_train_dense_annotations.json",
        metadata={"help": "Dense annotations for the 2,000-image train subset."},
    )
    dense_annotations_path: str = field(
        default="data/visdial_1.0_val_dense_annotations.json",
        metadata={"help": "Val dense annotations, used only for evaluation."},
    )
    loss: str = field(default="soft_ce", metadata={"help": "soft_ce | approx_ndcg"})
    dense_weight: float = field(default=1.0)
    sparse_weight: float = field(
        default=0.0,
        metadata={"help": "Weight on the original one-hot loss; >0 retains some MRR."},
    )
    approx_ndcg_temperature: float = field(default=1.0)
    max_eval_images: Optional[int] = field(default=None)


def main():
    parser = HfArgumentParser((Arguments, FGATrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    set_seed(training_args.seed)

    # The relevance vector is an extra column the model's forward does not take;
    # without this Trainer drops it and the dense term is silently always zero.
    training_args.remove_unused_columns = False

    params = load_visdial_params(args.visdial_params_path)
    vocab_size = vocab_size_from_params(params)

    def make_dataset(split, limit=None):
        return VisDialDataset(
            visdial_data_path=args.visdial_data_path,
            image_features_path=args.image_features_path,
            split=split,
            vocab_size=vocab_size,
            limit_images=limit,
            in_memory=False,
        )

    train_split = make_dataset("train")
    train_dataset = DenseVisDialDataset(
        dataset=train_split,
        dense_annotations_path=args.train_dense_path,
        image_ids=image_ids_from_params(params, "train"),
        keep_sparse_labels=True,
    )
    logger.info(f"{len(train_dataset)} densely annotated training rounds")

    eval_dataset = make_dataset("val", args.max_eval_images)
    val_image_ids = image_ids_from_params(params, "val")
    dense = load_dense_annotations(args.dense_annotations_path, val_image_ids)
    compute_metrics = build_compute_metrics(
        dense_relevance=dense["relevance"],
        dense_round_ids=dense["round_ids"],
        num_rounds=eval_dataset.n_qa_per_dial,
    )

    model = FGAForVisualDialog.from_pretrained(args.model_name_or_path)

    class Collator(VisDialCollator):
        """Carry the relevance vector through alongside the model inputs."""

        def __call__(self, features):
            relevance = [f.pop("relevance") for f in features] if "relevance" in features[0] else None
            batch = super().__call__(features)
            if relevance is not None:
                import numpy as np
                import torch

                batch["relevance"] = torch.from_numpy(np.stack(relevance))
            return batch

    trainer = DenseFinetuneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(return_labels=True),
        compute_metrics=compute_metrics,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        loss_kind=args.loss,
        approx_ndcg_temperature=args.approx_ndcg_temperature,
    )

    logger.info(f"Evaluating before finetuning ({args.loss}, sparse_weight={args.sparse_weight})")
    before = trainer.evaluate()
    trainer.log_metrics("eval_before", before)

    trainer.train()
    trainer.save_model()

    after = trainer.evaluate()
    trainer.log_metrics("eval_after", after)
    trainer.save_metrics("eval_after", after)

    print("\n=== dense finetuning summary ===")
    print(f"loss={args.loss} dense_weight={args.dense_weight} sparse_weight={args.sparse_weight}")
    for key in ("eval_ndcg", "eval_mrr", "eval_r1", "eval_r5", "eval_r10", "eval_mean_rank"):
        if key in before and key in after:
            print(f"  {key:<16} {before[key]:.4f} -> {after[key]:.4f}  ({after[key] - before[key]:+.4f})")


if __name__ == "__main__":
    main()
