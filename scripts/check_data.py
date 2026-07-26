#!/usr/bin/env python
"""Validate the VisDial files before queuing a long training job.

A 48-hour GPU allocation that dies at step 0 on a missing h5 key is an expensive
way to find a typo. This checks every assumption the pipeline makes -- file
presence, dataset keys, shapes, index ranges, vocabulary agreement, dense
annotation coverage -- and then runs two real optimizer steps and an evaluation
on a handful of images.

```bash
python scripts/check_data.py --image_features_path data/frcnn_features_new.h5
```

Exits 0 when everything is consistent, 1 otherwise, so it can gate an sbatch job.
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

OK = "  ok  "
BAD = " FAIL "
WARN = " warn "

REQUIRED_KEYS = (
    "ques_{split}",
    "ques_length_{split}",
    "ans_{split}",
    "ans_length_{split}",
    "opt_{split}",
    "opt_list_{split}",
    "opt_length_{split}",
    "cap_{split}",
    "cap_length_{split}",
)
LABELLED_ONLY = ("ans_index_{split}",)


class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def check(self, ok: bool, message: str, detail: str = "") -> bool:
        print(f"[{OK if ok else BAD}] {message}" + (f"  -- {detail}" if detail and not ok else ""))
        self.failures += not ok
        return ok

    def warn(self, ok: bool, message: str, detail: str = "") -> bool:
        if not ok:
            self.warnings += 1
            print(f"[{WARN}] {message}" + (f"  -- {detail}" if detail else ""))
        else:
            print(f"[{OK}] {message}")
        return ok

    def note(self, message: str) -> None:
        print(f"[      ] {message}")


def check_dialogs(report: Report, path: str, splits) -> dict:
    print(f"\n== dialogs: {path}")
    if not report.check(os.path.exists(path), f"{path} exists"):
        return {}

    counts = {}
    with h5py.File(path, "r") as h5:
        for split in splits:
            keys = [k.format(split=split) for k in REQUIRED_KEYS]
            if split != "test":
                keys += [k.format(split=split) for k in LABELLED_ONLY]
            missing = [k for k in keys if k not in h5]
            if not report.check(not missing, f"{split}: all datasets present", f"missing {missing}"):
                continue

            ques = h5[f"ques_{split}"]
            num_images, num_rounds, trunc = ques.shape
            counts[split] = {"images": num_images, "rounds": num_rounds, "trunc": trunc}
            report.note(f"{split}: {num_images} images x {num_rounds} rounds, truncated at {trunc} tokens")

            report.check(
                h5[f"ans_{split}"].shape == ques.shape,
                f"{split}: question and answer arrays agree",
                f"{h5[f'ans_{split}'].shape} vs {ques.shape}",
            )
            report.check(
                h5[f"opt_{split}"].shape == (num_images, num_rounds, 100),
                f"{split}: 100 options per round",
                str(h5[f"opt_{split}"].shape),
            )
            report.check(
                h5[f"cap_{split}"].shape[0] == num_images,
                f"{split}: one caption per image",
            )

            # Options are 1-based indices into the answer pool.
            pool = h5[f"opt_list_{split}"].shape[0]
            sample = h5[f"opt_{split}"][: min(64, num_images)]
            report.check(
                sample.min() >= 1 and sample.max() <= pool,
                f"{split}: option indices lie inside the answer pool of {pool}",
                f"range [{sample.min()}, {sample.max()}]",
            )

            if split != "test":
                idx = h5[f"ans_index_{split}"][: min(64, num_images)]
                report.check(
                    idx.min() >= 1 and idx.max() <= 100,
                    f"{split}: ground-truth index is 1..100",
                    f"range [{idx.min()}, {idx.max()}]",
                )

            lengths = h5[f"ques_length_{split}"][: min(64, num_images)]
            report.check(
                lengths.max() <= trunc,
                f"{split}: question lengths fit the truncation",
                f"max {lengths.max()} > {trunc}",
            )
    return counts


def check_params(report: Report, path: str, counts: dict) -> int:
    print(f"\n== vocabulary: {path}")
    if not report.check(os.path.exists(path), f"{path} exists"):
        return 0

    from fga.data import load_visdial_params, vocab_size_from_params

    params = load_visdial_params(path)
    vocab_size = vocab_size_from_params(params)
    report.note(f"{len(params['word2ind'])} words -> embedding table of {vocab_size} rows")

    report.check(
        max(params["word2ind"].values()) == len(params["word2ind"]),
        "word ids are contiguous from 1",
        f"max id {max(params['word2ind'].values())} for {len(params['word2ind'])} words",
    )

    for split, info in counts.items():
        key = f"unique_img_{split}"
        if not report.check(key in params, f"{key} present"):
            continue
        report.check(
            len(params[key]) == info["images"],
            f"{split}: filename list matches the dialog count",
            f"{len(params[key])} names vs {info['images']} images",
        )
    return vocab_size


def check_features(report: Report, path: str, counts: dict) -> None:
    print(f"\n== image features: {path}")
    if not report.check(os.path.exists(path), f"{path} exists", "this is the file most often missing"):
        return

    with h5py.File(path, "r") as h5:
        report.note(f"datasets: {sorted(h5.keys())}")
        dims = set()
        for split, info in counts.items():
            key = f"{split}_features"
            if not report.check(key in h5, f"{key} present", f"available: {sorted(h5.keys())}"):
                continue
            rows, regions, dim = h5[key].shape
            dims.add(dim)
            report.note(f"{split}: {rows} images x {regions} regions x {dim} dims")
            report.check(
                rows >= info["images"],
                f"{split}: features cover every image",
                f"{rows} rows for {info['images']} images",
            )
            sample = np.asarray(h5[key][0], dtype=np.float32)
            report.check(np.isfinite(sample).all(), f"{split}: features are finite")
            report.warn(
                np.abs(sample).sum() > 0,
                f"{split}: first image's features are non-zero",
                "all-zero features usually mean a failed extraction",
            )
        report.check(len(dims) <= 1, "feature dimension is consistent across splits", str(dims))


def check_dense(report: Report, path: str, params_path: str) -> None:
    print(f"\n== dense annotations: {path}")
    if not report.warn(os.path.exists(path), f"{path} exists", "NDCG will be skipped without it"):
        return
    if not os.path.exists(params_path):
        return

    from fga.data import image_ids_from_params, load_visdial_params
    from fga.trainer import load_dense_annotations

    params = load_visdial_params(params_path)
    image_ids = image_ids_from_params(params, "val")
    dense = load_dense_annotations(path, image_ids)
    covered = int((dense["relevance"].sum(axis=1) > 0).sum())
    report.check(
        covered > 0.9 * len(image_ids),
        "dense annotations cover the val split",
        f"only {covered}/{len(image_ids)} images matched",
    )
    report.note(f"{covered}/{len(image_ids)} val images carry relevance judgements")


def smoke_train(report: Report, args, vocab_size: int) -> None:
    """Two optimizer steps and an evaluation, to catch anything static checks miss."""
    print("\n== two-step training smoke test")
    import torch

    from fga import FGAConfig, FGAForVisualDialog, VisDialCollator, VisDialDataset

    try:
        dataset = VisDialDataset(
            visdial_data_path=args.visdial_data_path,
            image_features_path=args.image_features_path,
            split="val",
            vocab_size=vocab_size,
            limit_images=4,
        )
        regions, dim = dataset.image_feature_shape
        config = FGAConfig(
            vocab_size=vocab_size,
            word_embed_dim=16,
            hidden_ques_dim=32,
            hidden_ans_dim=32,
            hidden_hist_dim=8,
            hidden_cap_dim=8,
            hidden_img_dim=dim,
            num_history_rounds=dataset.n_qa_per_dial - 1,
            utility_sizes=[100, 21, 41, regions, 21, 21],
            sharing_factor_weights={
                4: (dataset.n_qa_per_dial - 1, [0, 1]),
                5: (dataset.n_qa_per_dial - 1, [0, 1]),
            },
        )
        model = FGAForVisualDialog(config).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        collate = VisDialCollator()

        losses = []
        for step in range(2):
            batch = collate([dataset[i] for i in range(step * 4, step * 4 + 4)])
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss))

        report.check(all(np.isfinite(losses)), "loss is finite", str(losses))
        report.note(f"losses: {[round(loss, 4) for loss in losses]}")

        model.eval()
        with torch.no_grad():
            logits = model(**collate([dataset[i] for i in range(4)])).logits
        report.check(logits.shape == (4, 100), "eval produces 100 scores per round", str(tuple(logits.shape)))
        report.check(torch.isfinite(logits).all(), "eval scores are finite")
    except Exception as exc:  # noqa: BLE001 - the whole point is to surface it early
        report.check(False, "smoke test ran", f"{type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visdial_data_path", default="data/visdial_data.h5")
    parser.add_argument("--visdial_params_path", default="data/visdial_params.json")
    parser.add_argument("--image_features_path", default="data/frcnn_features_new.h5")
    parser.add_argument("--dense_annotations_path", default="data/visdial_1.0_val_dense_annotations.json")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--skip_smoke_test", action="store_true")
    args = parser.parse_args()

    report = Report()
    counts = check_dialogs(report, args.visdial_data_path, args.splits)
    vocab_size = check_params(report, args.visdial_params_path, counts)
    check_features(report, args.image_features_path, counts)
    check_dense(report, args.dense_annotations_path, args.visdial_params_path)

    if not args.skip_smoke_test and vocab_size and report.failures == 0:
        smoke_train(report, args, vocab_size)
    elif report.failures:
        print("\n== smoke test skipped: fix the failures above first")

    print()
    if report.failures:
        print(f"{report.failures} check(s) FAILED, {report.warnings} warning(s). Do not queue the job yet.")
        return 1
    print(f"All checks passed ({report.warnings} warning(s)). Safe to queue training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
