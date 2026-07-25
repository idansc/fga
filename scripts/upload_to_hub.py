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


def upload_model(api, repo_id: str, private: bool, dry_run: bool) -> None:
    """Publish the architecture and card. Deliberately uploads no weights."""
    from huggingface_hub import CommitOperationAdd

    from fga import FGAConfig

    config = FGAConfig(vocab_size=11322, hidden_img_dim=2048)
    config_json = os.path.join(HUB_DIR, "model", "config.json")
    config.to_json_file(config_json)

    card = os.path.join(HUB_DIR, "model", "README.md")
    print(f"Model repo: {repo_id} ({'private' if private else 'PUBLIC'})")
    print("  README.md")
    print("  config.json")
    print("  NOTE: no weights are uploaded; none exist. The card says so explicitly.")
    if dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd("README.md", card),
            CommitOperationAdd("config.json", config_json),
        ],
        commit_message="Add Factor Graph Attention architecture and model card",
    )
    print(f"\nhttps://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", choices=["dataset", "model", "both"], default="both")
    parser.add_argument("--dataset_repo", default="idansc/visdial-fga-preprocessed")
    parser.add_argument("--model_repo", default="idansc/fga")
    parser.add_argument("--data_dir", default="data")
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
        upload_model(api, args.model_repo, private, args.dry_run)


if __name__ == "__main__":
    main()
