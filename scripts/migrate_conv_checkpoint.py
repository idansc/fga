#!/usr/bin/env python
"""Convert a checkpoint whose attention projections were 1x1 convolutions.

The attention's projections are linear maps and are now stored as `nn.Linear`.
Earlier checkpoints wrote them as `Conv1d(kernel_size=1)` -- the same numbers with
a trailing singleton axis on every weight, `(out, in, 1)` instead of `(out, in)`.
This rewrites the file once, in place or to a new directory, rather than the
model carrying a load-time shim forever.

```bash
python scripts/migrate_conv_checkpoint.py models/fga-hub            # in place
python scripts/migrate_conv_checkpoint.py old-dir --output new-dir
```

Idempotent: a checkpoint already in the new shape is left untouched.
"""

import argparse
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

#: Only weights under these module names are eligible; embeddings, LSTMs and
#: heads never had the convolutional shape.
ATTENTION_MARKERS = ("mul_atten.", "attention.", "un_models.", "pp_models.", "tri_models.", "reduce_potentials.")


def migrate(path: str, output: str) -> int:
    tensors, changed = {}, 0
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if (
                key.endswith(".weight")
                and tensor.dim() == 3
                and tensor.size(-1) == 1
                and any(marker in key for marker in ATTENTION_MARKERS)
            ):
                tensor = tensor.squeeze(-1)
                changed += 1
            tensors[key] = tensor
    save_file(tensors, output, metadata={"format": "pt"})
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_dir", help="Directory holding model.safetensors (+ config.json).")
    parser.add_argument("--output", default=None, help="Write here instead of migrating in place.")
    args = parser.parse_args()

    source = os.path.join(args.checkpoint_dir, "model.safetensors")
    if not os.path.exists(source):
        raise SystemExit(f"{source} not found")

    out_dir = args.output or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.output:
        for name in os.listdir(args.checkpoint_dir):
            if name != "model.safetensors":
                shutil.copy2(os.path.join(args.checkpoint_dir, name), os.path.join(out_dir, name))

    changed = migrate(source, os.path.join(out_dir, "model.safetensors"))
    print(f"{changed} weights reshaped -> {out_dir}" if changed else "already in the new shape; nothing to do")


if __name__ == "__main__":
    sys.exit(main())
