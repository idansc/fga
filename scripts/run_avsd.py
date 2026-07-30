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
from fga.tasks.video_dialog.data import AVSDCollator, AVSDDataset, load_vocab, render  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    data: str = field(default="avsd/avsd.h5")
    # Defaults are the released configuration; see AVSDGenerationConfig.
    hidden_size: int = field(default=256, metadata={"help": "Stream projection and decoder width."})
    question_dim: int = field(default=256, metadata={"help": "Question LSTM width."})
    history_dim: int = field(default=128)
    embed_size: int = field(default=128)
    decoder_proj: int = field(default=128)
    decoder_layers: int = field(default=1)
    dropout: float = field(default=0.5)
    tie_embeddings: bool = field(
        default=False,
        metadata={"help": "Share one word embedding table; the original builds three."},
    )
    beam_width: int = field(default=3, metadata={"help": "0 for greedy decoding."})
    length_penalty: float = field(default=1.0)
    features_in_memory: bool = field(default=False, metadata={"help": "Video table is large; check RAM."})
    ablate: str = field(
        default="none",
        metadata={"help": "Zero a modality to see what it was worth: none, video, audio or av."},
    )
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
    # Frames are present only when the h5 was prepared with --spatial_dir.
    num_frames, frame_regions, frame_dim = (
        sample["frame_features"].shape if "frame_features" in sample else (0, 49, 512)
    )
    logger.info(
        f"{len(words)} words | video {streams}x{segments}x{video_dim} | audio {audio_steps}x{audio_dim} | "
        + (f"frames {num_frames}x{frame_regions}x{frame_dim} | " if num_frames else "no spatial frames | ")
        + f"train {len(train) if train else 0} turns, val {len(val)}"
    )

    model = AVSDForResponseGeneration(
        AVSDGenerationConfig(
            num_video_frames=num_frames,
            num_frame_regions=frame_regions,
            frame_dim=frame_dim,
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
            embed_size=args.embed_size,
            decoder_proj=args.decoder_proj,
            tie_embeddings=args.tie_embeddings,
            dropout=args.dropout,
        )
    )
    logger.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    blanked = {
        "none": (),
        "video": ("video_features", "frame_features"),
        "audio": ("audio_features",),
        "av": ("video_features", "frame_features", "audio_features"),
    }[args.ablate]
    if blanked:
        logger.info(f"ablation: zeroing {', '.join(blanked)}")
    collator = AVSDCollator(zero=blanked)

    # answer_input_ids is both the teacher-forcing input and the label, so it has
    # to survive the collator.
    training_args.remove_unused_columns = False
    training_args.label_names = ["answer_input_ids"]
    # Perplexity is a function of the loss alone. Asking for it through
    # compute_metrics would make the Trainer accumulate every logit tensor over
    # the validation set -- 17,870 turns x 20 tokens x 6,055 words is about 8.6 GB
    # of float32, which is an out-of-memory error rather than a metric.
    training_args.prediction_loss_only = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_perplexity"] = math.exp(min(metrics["eval_loss"], 20))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.generate_samples:
        model.eval().to(training_args.device)
        collate = collator  # generate under the same ablation the model was trained with
        rows = [val[i] for i in range(min(args.generate_samples, len(val)))]
        written = []
        for start in range(0, len(rows), 32):
            batch = {k: v.to(training_args.device) for k, v in collate(rows[start : start + 32]).items()}
            inputs = (
                batch["question_input_ids"], batch["history_input_ids"],
                batch["video_features"], batch["audio_features"], batch.get("frame_features"),
            )
            produced = (
                model.beam_search(*inputs, beam_width=args.beam_width, length_penalty=args.length_penalty)
                if args.beam_width else model.generate_answers(*inputs)
            ).cpu().numpy()
            for i, ids in enumerate(produced):
                written.append(
                    {
                        "question": render(rows[start + i]["question_input_ids"], words),
                        "reference": render(rows[start + i]["answer_input_ids"], words),
                        "generated": render(ids, words),
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
