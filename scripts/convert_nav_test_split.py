#!/usr/bin/env python
"""Convert the original's fixed test episodes into a plain JSON file.

Target-driven navigation is scored on a fixed set of episodes, not on random
ones: `test_val_split/*_test.pkl` in barmayo/spatial_attention pins the scene, the
target object *instance* and the agent's start pose for each of 3,914 episodes.
Comparing against the published numbers means running exactly those.

They are Python pickles of the original's own classes, which import the AI2-THOR
simulator at module load, and they carry a GloVe tensor per episode that is
already available from `thor_glove`. This rewrites them as JSON holding only what
an episode needs, so nothing downstream needs pickle, torch or the simulator.

```bash
python scripts/convert_nav_test_split.py \
    --split_dir spatial_attention/test_val_split --output nav/data/test_episodes.json
```
"""

import argparse
import glob
import json
import os
import pickle
import sys
import types


def stub_simulator():
    """Let the pickles resolve without AI2-THOR installed."""
    for name in ("ai2thor", "ai2thor.controller", "ai2thor.util", "ai2thor.util.metrics"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["ai2thor.controller"].Controller = object
    sys.modules["ai2thor.controller"].distance = lambda *a, **k: 0
    sys.modules["ai2thor.util.metrics"].get_shortest_path_to_object = lambda *a, **k: None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split_dir", required=True, help="Directory holding *_test.pkl.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()

    stub_simulator()
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.split_dir.rstrip("/"))))

    episodes = []
    paths = sorted(glob.glob(os.path.join(args.split_dir, f"*_{args.split}.pkl")))
    # `*_test_val.pkl` is a combined file and matches the val glob; the splits are
    # the four per-room files, and mixing the combined one in double-counts.
    paths = [p for p in paths if not os.path.basename(p).endswith("_test_val.pkl")]
    for path in paths:
        with open(path, "rb") as handle:
            loaded = pickle.load(handle)
        for item in loaded:
            episodes.append(
                {
                    "scene": item["scene"],
                    "target": item["goal_object_type"],
                    # A ThorAgentState, whose str() is the "x|z|rotation|horizon"
                    # pose string the cached feature maps are keyed by.
                    "state": str(item["state"]),
                    # The specific instances that count as found. Accepting any
                    # instance of the class would make the task easier than the
                    # one the published numbers were measured on.
                    "task_data": list(item["task_data"]),
                }
            )
        print(f"{os.path.basename(path)}: {len(loaded)}")

    with open(args.output, "w") as handle:
        json.dump(episodes, handle)
    print(f"wrote {args.output}: {len(episodes)} episodes")


if __name__ == "__main__":
    sys.exit(main())
