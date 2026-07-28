#!/usr/bin/env python
"""Train answer generation on DSTC7 Audio-Visual Scene-Aware Dialog.

The model attends the question over the video and audio streams, and a decoder
writes the answer. Perplexity is reported during training; the challenge's own
metrics (BLEU, METEOR, ROUGE-L, CIDEr) come from the organisers' `dstc7avsd_eval`
package run against the generated file, rather than from a reimplementation here.

The features DSTC7 ships are pooled over space, so attention here is over
*moments* -- when in the video the answer lives. Per-frame spatial attention needs
conv maps extracted from the Charades videos.

```bash
python scripts/run_avsd.py --data avsd/avsd.h5 --output_dir models/avsd \
    --do_train --do_eval --per_device_train_batch_size 64 --num_train_epochs 15 --bf16
```
"""

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.video_dialog import AVSDForResponseGeneration, AVSDGenerationConfig  # noqa: E402
from fga.tasks.video_dialog.data import AVSDCollator, AVSDDataset, load_vocab  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    data: str = field(default="avsd/avsd.h5")
    hidden_size: int = field(default=512)
    question_dim: int = field(default=256)
    history_dim: int = field(default=256)
    decoder_layers: int = field(default=1)
    dropout: float = field(default=0.5)
    features_in_memory: bool = field(default=False, metadata={"help": "Video table is large; check RAM."})
    generate_samples: int = field(default=200, metadata={"help": "Answers written out after evaluation."})


def main():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format="%(message)s")
    set_seed(training_args.seed)

    words = load_vocab(args.data)
    train = AVSDDataset(args.data, "train", args.features_in_memory) if training_args.do_train else None
    val = AVSDDataset(args.data, "val", False)
    sample = val[0]
    streams, segments, video_dim = sample["video_features"].shape
    audio_steps, audio_dim = sample["audio_features"].shape
    logger.info(
        f"{len(words)} words | video {streams}x{segments}x{video_dim} | audio {audio_steps}x{audio_dim} | "
        f"train {len(train) if train else 0} turns, val {len(val)}"
    )

    model = AVSDForResponseGeneration(
        AVSDGenerationConfig(
            vocab_size=len(words) + 1,
            max_answer_length=sample["answer_input_ids"].shape[0],
            decoder_layers=args.decoder_layers,
            question_dim=args.question_dim,
            video_dim=video_dim,
            audio_dim=audio_dim,
            hidden_size=args.hidden_size,
            num_video_streams=streams,
            num_video_regions=segments,
            num_audio_steps=audio_steps,
            max_question_length=sample["question_input_ids"].shape[0],
            history_dim=args.history_dim,
            dropout=args.dropout,
        )
    )
    logger.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    def compute_metrics(prediction):
        # Perplexity over the answer tokens, padding excluded, which is the
        # trainable signal; the challenge metrics are computed separately from
        # generated text by the organisers' package.
        logits = prediction.predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        labels = np.asarray(prediction.label_ids)
        logits = torch.as_tensor(np.asarray(logits, dtype=np.float32))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            torch.as_tensor(labels).reshape(-1),
            ignore_index=0,
        )
        return {"perplexity": float(math.exp(min(float(loss), 20)))}

    # answer_input_ids is both the teacher-forcing input and the label, so it must
    # survive the collator and be visible to the metric.
    training_args.remove_unused_columns = False
    training_args.label_names = ["answer_input_ids"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=AVSDCollator(),
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.generate_samples:
        model.eval().to(training_args.device)
        collate = AVSDCollator()
        rows = [val[i] for i in range(min(args.generate_samples, len(val)))]
        written = []
        for start in range(0, len(rows), 32):
            batch = {k: v.to(training_args.device) for k, v in collate(rows[start : start + 32]).items()}
            produced = model.generate_answers(
                batch["question_input_ids"], batch["history_input_ids"],
                batch["video_features"], batch["audio_features"],
            ).cpu().numpy()
            for i, ids in enumerate(produced):
                def render(sequence):
                    return " ".join(words[t - 1] for t in sequence if t > 0)
                written.append(
                    {
                        "question": render(rows[start + i]["question_input_ids"]),
                        "reference": render(rows[start + i]["answer_input_ids"]),
                        "generated": render(ids),
                    }
                )
        path = os.path.join(training_args.output_dir, "generated.json")
        with open(path, "w") as handle:
            json.dump(written, handle, indent=1)
        logger.info(f"wrote {len(written)} generated answers to {path}")
        for item in written[:3]:
            logger.info(f"  Q: {item['question']}\n    ref: {item['reference']}\n    gen: {item['generated']}")


if __name__ == "__main__":
    main()
