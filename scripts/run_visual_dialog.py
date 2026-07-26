#!/usr/bin/env python
"""Train or evaluate Factor Graph Attention on VisDial v1.0.

Examples:

```bash
# train
python scripts/run_visual_dialog.py \
    --image_features_path data/frcnn_features_new.h5 \
    --output_dir models/baseline --do_train --do_eval \
    --per_device_train_batch_size 128 --learning_rate 1e-3 --num_train_epochs 10

# evaluate a trained checkpoint on val
python scripts/run_visual_dialog.py \
    --image_features_path data/frcnn_features_new.h5 \
    --model_name_or_path models/baseline --output_dir models/baseline \
    --do_eval --write_submission

# produce a test-split submission for EvalAI
python scripts/run_visual_dialog.py \
    --image_features_path data/frcnn_features_new.h5 \
    --model_name_or_path models/baseline --output_dir models/baseline \
    --do_predict --write_submission
```
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import EarlyStoppingCallback, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga import FGAConfig, FGAForVisualDialog, VisDialCollator, VisDialDataset  # noqa: E402
from fga.data import image_ids_from_params, load_visdial_params, vocab_size_from_params  # noqa: E402
from fga.metrics import build_compute_metrics  # noqa: E402
from fga.trainer import FGATrainer, FGATrainingArguments, load_dense_annotations  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Which model to train, and how it is shaped."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Directory or Hub id of a trained FGA model. Omit to train from scratch."},
    )
    word_embed_dim: int = field(default=200, metadata={"help": "Word embedding dimension."})
    hidden_ques_dim: int = field(default=512, metadata={"help": "Question LSTM hidden size."})
    hidden_ans_dim: int = field(default=512, metadata={"help": "Answer LSTM hidden size."})
    hidden_hist_dim: int = field(default=128, metadata={"help": "History LSTM hidden size."})
    hidden_cap_dim: int = field(default=128, metadata={"help": "Caption LSTM hidden size."})
    initializer_type: str = field(default="he", metadata={"help": "he | xavier | default."})
    lstm_initializer_type: str = field(default="he", metadata={"help": "he | xavier | default."})


@dataclass
class DataArguments:
    """Where the preprocessed VisDial files live."""

    visdial_data_path: str = field(default="data/visdial_data.h5", metadata={"help": "Preprocessed dialogs."})
    visdial_params_path: str = field(default="data/visdial_params.json", metadata={"help": "Vocabulary."})
    image_features_path: str = field(
        default="data/frcnn_features_new.h5",
        metadata={"help": "h5 with {split}_features of shape (num_images, num_regions, feature_dim)."},
    )
    dense_annotations_path: str = field(
        default="data/visdial_1.0_val_dense_annotations.json",
        metadata={"help": "Val dense relevance annotations; required for NDCG."},
    )
    trunc_length: int = field(default=20, metadata={"help": "Question/answer truncation length."})
    caption_length: int = field(default=40, metadata={"help": "Caption truncation length."})
    max_train_images: Optional[int] = field(default=None, metadata={"help": "Debug: cap the training images."})
    max_eval_images: Optional[int] = field(default=None, metadata={"help": "Debug: cap the eval images."})
    max_predict_images: Optional[int] = field(default=None, metadata={"help": "Debug: cap the test images."})
    image_features_in_memory: bool = field(
        default=True,
        metadata={"help": "Load image features into RAM. Set False to stream them from disk instead."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, FGATrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=training_args.get_process_log_level(),
    )
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    logger.warning(
        f"Process rank: {training_args.process_index}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.parallel_mode.value == 'distributed')}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    set_seed(training_args.seed)

    params = load_visdial_params(data_args.visdial_params_path)
    vocab_size = vocab_size_from_params(params)

    def make_dataset(split: str, limit: Optional[int]) -> VisDialDataset:
        return VisDialDataset(
            visdial_data_path=data_args.visdial_data_path,
            image_features_path=data_args.image_features_path,
            split=split,
            vocab_size=vocab_size,
            trunc_length=data_args.trunc_length,
            caption_length=data_args.caption_length,
            limit_images=limit,
            in_memory=data_args.image_features_in_memory,
        )

    train_dataset = make_dataset("train", data_args.max_train_images) if training_args.do_train else None
    eval_dataset = make_dataset("val", data_args.max_eval_images) if training_args.do_eval else None
    predict_dataset = make_dataset("test", data_args.max_predict_images) if training_args.do_predict else None

    reference = train_dataset or eval_dataset or predict_dataset
    if reference is None:
        raise ValueError("Nothing to do: pass at least one of --do_train, --do_eval, --do_predict.")
    image_feature_dim = reference.image_feature_shape[-1]
    num_regions = reference.image_feature_shape[0]

    if model_args.model_name_or_path:
        model = FGAForVisualDialog.from_pretrained(model_args.model_name_or_path)
        logger.info(f"Loaded model from {model_args.model_name_or_path}")
    else:
        config = FGAConfig(
            vocab_size=vocab_size,
            word_embed_dim=model_args.word_embed_dim,
            hidden_ques_dim=model_args.hidden_ques_dim,
            hidden_ans_dim=model_args.hidden_ans_dim,
            hidden_hist_dim=model_args.hidden_hist_dim,
            hidden_cap_dim=model_args.hidden_cap_dim,
            hidden_img_dim=image_feature_dim,
            num_history_rounds=reference.n_qa_per_dial - 1,
            utility_sizes=[
                100,
                data_args.trunc_length + 1,
                data_args.caption_length + 1,
                num_regions,
                data_args.trunc_length + 1,
                data_args.trunc_length + 1,
            ],
            sharing_factor_weights={
                4: (reference.n_qa_per_dial - 1, [0, 1]),
                5: (reference.n_qa_per_dial - 1, [0, 1]),
            },
            initializer_type=model_args.initializer_type,
            lstm_initializer_type=model_args.lstm_initializer_type,
        )
        model = FGAForVisualDialog(config)
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # NDCG needs the dense relevance judgements aligned to the val image order.
    compute_metrics = None
    val_image_ids = image_ids_from_params(params, "val")
    if eval_dataset is not None:
        if os.path.exists(data_args.dense_annotations_path):
            dense = load_dense_annotations(data_args.dense_annotations_path, val_image_ids)
            compute_metrics = build_compute_metrics(
                dense_relevance=dense["relevance"],
                dense_round_ids=dense["round_ids"],
                num_rounds=eval_dataset.n_qa_per_dial,
            )
        else:
            logger.warning(f"{data_args.dense_annotations_path} not found; reporting sparse metrics only.")
            compute_metrics = build_compute_metrics(num_rounds=eval_dataset.n_qa_per_dial)

    trainer = FGATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VisDialCollator(),
        compute_metrics=compute_metrics,
        image_ids=val_image_ids,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if training_args.load_best_model_at_end else None,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint
        # `overwrite_output_dir` exists on transformers 4.x and was dropped in 5.x.
        overwrite = getattr(training_args, "overwrite_output_dir", False)
        if checkpoint is None and os.path.isdir(training_args.output_dir) and not overwrite:
            checkpoint = get_last_checkpoint(training_args.output_dir)
            if checkpoint is not None:
                logger.info(f"Resuming from {checkpoint}")
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if training_args.write_submission:
            output = trainer.predict(eval_dataset)
            trainer.write_submission(output.predictions, filename="submission_val.json")

    if training_args.do_predict:
        trainer.image_ids = image_ids_from_params(params, "test")
        trainer.num_rounds_per_image = predict_dataset.num_rounds_per_image
        output = trainer.predict(predict_dataset, metric_key_prefix="predict")
        if training_args.write_submission:
            trainer.write_submission(output.predictions, filename="submission_test.json")


if __name__ == "__main__":
    main()
