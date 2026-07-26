#!/usr/bin/env python
"""Convert an original FGA `.pth.tar` checkpoint into a HuggingFace model folder.

The original release saved `{"model": state_dict, "optimizer": <optimizer object>,
"args": <argparse.Namespace>, "epoch": int}` via `torch.save`. Because the
optimizer and args were pickled as live Python objects, such a file can only be
loaded with `weights_only=False`, i.e. by executing arbitrary pickle opcodes --
only run this on checkpoints you trust.

The output is a directory with `config.json` and `model.safetensors`, loadable
with `FGAForVisualDialog.from_pretrained(...)`.

Example:

```bash
python scripts/convert_legacy_checkpoint.py \
    --checkpoint models/baseline/best_model_mrr.pth.tar \
    --visdial_params data/visdial_params.json \
    --image_feature_dim 2048 --output_dir models/fga-frcnn-hf
```
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga import FGAConfig, FGAForVisualDialog  # noqa: E402
from fga.data import load_visdial_params, vocab_size_from_params  # noqa: E402

# Old top-level module name -> new location inside FGAForVisualDialog.
_PREFIX_RENAMES = (
    ("word_embedddings.", "fga.text_encoder.word_embeddings."),  # sic: three d's in the original
    ("lstm_ques.", "fga.text_encoder.lstms.question."),
    ("lstm_ans.", "fga.text_encoder.lstms.answer."),
    ("lstm_hist_cap.", "fga.text_encoder.lstms.caption."),
    ("lstm_hist_ques.", "fga.text_encoder.lstms.history_question."),
    ("lstm_hist_ans.", "fga.text_encoder.lstms.history_answer."),
    ("qahistnet.", "fga.qahistnet."),
    ("mul_atten.", "fga.mul_atten."),
    ("simnet.", "simnet."),
)


# `pp_models` used to be keyed `"0"` (self-interaction) and `"(0, 1)"` (a pair).
# Those are not valid attribute-path segments, so they are now `self_0` and `0_1`.
_SELF_FACTOR_RE = re.compile(r"(pp_models\.)(\d+)\.")
_PAIR_FACTOR_RE = re.compile(r"(pp_models\.)\((\d+),\s*(\d+)\)\.")


def rename_factor_keys(name: str) -> str:
    """Translate legacy `pp_models` keys to the attribute-safe ones."""
    name = _PAIR_FACTOR_RE.sub(r"\g<1>\g<2>_\g<3>.", name)
    return _SELF_FACTOR_RE.sub(r"\g<1>self_\g<2>.", name)


def remap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], list, list]:
    """Rename legacy parameter keys to the current module layout.

    Returns the remapped state dict, the keys that matched no known module, and
    the keys that were deliberately dropped.
    """
    remapped: Dict[str, torch.Tensor] = {}
    unmatched = []
    dropped = []
    for key, value in state_dict.items():
        # Strip the wrapper added by nn.DataParallel.
        name = re.sub(r"^module\.", "", key)
        name = rename_factor_keys(name)

        # Self-interaction factors only ever consumed the X marginal; the Y one
        # was computed and thrown away, so it never trained. Drop it.
        if re.search(r"pp_models\.self_\d+\.margin_Y\.", name):
            dropped.append(key)
            continue

        # 1x1-convolution weights become Linear weights: drop the trailing axis.
        if name.endswith(".weight") and value.dim() == 3 and value.size(-1) == 1 and "lstm" not in name:
            value = value.squeeze(-1)

        for old, new in _PREFIX_RENAMES:
            if name.startswith(old):
                remapped[new + name[len(old) :]] = value
                break
        else:
            unmatched.append(key)
    return remapped, unmatched, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model_mrr.pth.tar")
    parser.add_argument("--output_dir", required=True, help="Where to write config.json + model.safetensors")
    parser.add_argument("--visdial_params", default="data/visdial_params.json", help="To recover the vocabulary size")
    parser.add_argument("--image_feature_dim", type=int, default=2048, help="2048 for F-RCNN, 512 for VGG")
    parser.add_argument("--num_regions", type=int, default=37, help="Image regions per image")
    parser.add_argument("--trunc_length", type=int, default=20, help="Question/answer truncation length")
    parser.add_argument("--caption_length", type=int, default=40, help="Caption truncation length")
    parser.add_argument("--num_history_rounds", type=int, default=9)
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push to, e.g. 'idansc/fga'.")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint} (weights_only=False; the file pickles an optimizer and an argparse Namespace)")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict, got {type(state_dict)}. Was this checkpoint saved with torch.save?")

    legacy_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None

    def from_legacy(name: str, default):
        return getattr(legacy_args, name, default) if legacy_args is not None else default

    remapped, unmatched, dropped = remap_state_dict(state_dict)
    if unmatched:
        raise KeyError(f"These checkpoint keys did not match any known module: {unmatched}")
    if dropped:
        print(f"Dropped {len(dropped)} untrained self-interaction margin_Y tensors (they never received a gradient).")

    embedding = remapped["fga.text_encoder.word_embeddings.weight"]
    vocab_size = embedding.shape[0]
    word_embed_dim = embedding.shape[1]

    if os.path.exists(args.visdial_params):
        expected = vocab_size_from_params(load_visdial_params(args.visdial_params))
        if expected != vocab_size:
            print(f"Warning: checkpoint has {vocab_size} embedding rows, {args.visdial_params} implies {expected}.")

    config = FGAConfig(
        vocab_size=vocab_size,
        word_embed_dim=word_embed_dim,
        hidden_ques_dim=remapped["fga.text_encoder.lstms.question.weight_hh_l0"].shape[1],
        hidden_ans_dim=remapped["fga.text_encoder.lstms.answer.weight_hh_l0"].shape[1],
        hidden_hist_dim=remapped["fga.text_encoder.lstms.history_question.weight_hh_l0"].shape[1],
        hidden_cap_dim=remapped["fga.text_encoder.lstms.caption.weight_hh_l0"].shape[1],
        hidden_img_dim=args.image_feature_dim,
        num_history_rounds=args.num_history_rounds,
        utility_sizes=[
            100,
            args.trunc_length + 1,
            args.caption_length + 1,
            args.num_regions,
            args.trunc_length + 1,
            args.trunc_length + 1,
        ],
        sharing_factor_weights={
            4: (args.num_history_rounds, [0, 1]),
            5: (args.num_history_rounds, [0, 1]),
        },
        initializer_type=from_legacy("initialization", "he") or "he",
        lstm_initializer_type=from_legacy("lstm_initialization", "he") or "he",
    )

    model = FGAForVisualDialog(config)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "State dict did not line up with the model.\n"
            f"  missing: {missing}\n  unexpected: {unexpected}\n"
            "Check --image_feature_dim / --num_regions / --num_history_rounds."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    metadata = {
        "source_checkpoint": os.path.basename(args.checkpoint),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "metrics": {k: float(v) for k, v in (checkpoint.get("metrics") or {}).items()}
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("metrics"), dict)
        else None,
    }
    with open(os.path.join(args.output_dir, "conversion_info.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {args.output_dir} ({sum(p.numel() for p in model.parameters()):,} params)")
    if metadata["metrics"]:
        print(f"Original metrics recorded in the checkpoint: {metadata['metrics']}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
