#!/usr/bin/env python
"""Publish the FGA dataset and model repositories to the HuggingFace Hub.

Authentication comes from your stored Hub login, so no token appears in this file,
in the command line, or in shell history. Log in once:

```bash
hf auth login
```

Then:

```bash
python scripts/upload_to_hub.py --what dataset --dry-run   # inspect first
python scripts/upload_to_hub.py --what dataset
python scripts/upload_to_hub.py --what model
```

Repositories are created private by default; pass `--public` when you are ready for
them to be visible.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

HUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hub")

DATASET_FILES = (
    "visdial_data.h5",
    "visdial_params.json",
    "visdial_1.0_val_dense_annotations.json",
)


def resolve_dataset_files(data_dir: str):
    """Locate the dataset payload, following symlinks to their real targets."""
    resolved, missing = [], []
    for name in DATASET_FILES:
        path = os.path.realpath(os.path.join(data_dir, name))
        (resolved if os.path.exists(path) else missing).append((name, path))
    return resolved, missing


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024


def upload_dataset(api, repo_id: str, data_dir: str, private: bool, dry_run: bool) -> None:
    from huggingface_hub import CommitOperationAdd

    resolved, missing = resolve_dataset_files(data_dir)
    if missing:
        raise SystemExit(
            "Missing dataset files:\n"
            + "\n".join(f"  {name} (looked for {path})" for name, path in missing)
            + f"\nPut them in {data_dir} or pass --data_dir."
        )

    card = os.path.join(HUB_DIR, "dataset", "README.md")
    operations = [CommitOperationAdd("README.md", card)]
    operations += [CommitOperationAdd(name, path) for name, path in resolved]

    total = sum(os.path.getsize(path) for _, path in resolved)
    print(f"Dataset repo: {repo_id} ({'private' if private else 'PUBLIC'})")
    for name, path in resolved:
        print(f"  {name:<45} {human(os.path.getsize(path)):>8}")
    print(f"  {'README.md':<45} {human(os.path.getsize(card)):>8}")
    print(f"  total payload: {human(total)}")
    if dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Add preprocessed VisDial v1.0 files for Factor Graph Attention",
    )
    print(f"\nhttps://huggingface.co/datasets/{repo_id}")


def upload_model(api, repo_id: str, model_dir: str, private: bool, dry_run: bool) -> None:
    """Publish a trained model directory: weights, config and card.

    `model_dir` should be what `save_pretrained` produced — `model.safetensors`
    plus `config.json` — with the card copied alongside. Trainer checkpoints also
    carry `optimizer.pt` and RNG state, which are training-resumption artefacts
    and are deliberately not published.
    """
    from huggingface_hub import CommitOperationAdd

    required = ["config.json", "model.safetensors"]
    missing = [name for name in required if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        raise SystemExit(f"{model_dir} is missing {missing}; point --model_dir at a save_pretrained output.")

    card = os.path.join(model_dir, "README.md")
    if not os.path.exists(card):
        card = os.path.join(HUB_DIR, "model", "README.md")

    operations = [CommitOperationAdd("README.md", card)]
    operations += [CommitOperationAdd(name, os.path.join(model_dir, name)) for name in required]

    print(f"Model repo: {repo_id} ({'private' if private else 'PUBLIC'})")
    for op in operations:
        size = os.path.getsize(op.path_or_fileobj)
        print(f"  {op.path_in_repo:<24} {human(size):>9}")
    if dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Add Factor Graph Attention weights (VisDial v1.0, MRR 66.01)",
    )
    print(f"\nhttps://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", choices=["dataset", "model", "both"], default="both")
    parser.add_argument("--dataset_repo", default="idansc/visdial-fga-preprocessed")
    parser.add_argument("--model_repo", default="idansc/fga")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--model_dir",
        default="hub/model",
        help="save_pretrained output to publish: config.json + model.safetensors (+ README.md).",
    )
    parser.add_argument("--public", action="store_true", help="Create the repo public rather than private.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded and stop.")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    if not args.dry_run:
        try:
            user = api.whoami()["name"]
        except Exception:
            raise SystemExit("Not logged in. Run `hf auth login` first (do not paste a token here).") from None
        print(f"Logged in as {user}\n")

    private = not args.public
    if args.what in ("dataset", "both"):
        upload_dataset(api, args.dataset_repo, args.data_dir, private, args.dry_run)
    if args.what in ("model", "both"):
        print()
        upload_model(api, args.model_repo, args.model_dir, private, args.dry_run)


if __name__ == "__main__":
    main()
