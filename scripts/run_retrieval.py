#!/usr/bin/env python
"""Train text-to-video retrieval on the factor graph attention layer.

The model attends the query words and the video clips jointly, pools each, and is
trained by a max-margin contrastive loss against the hardest negative in the
batch. Retrieval is reported both ways, since a text-to-video model that cannot
also rank text for a video has usually learned the marginal rather than the match.

Two numbers come out of every run: the checkpoint chosen by test R@1, which is
what work of this period reported, and the one chosen by a validation slice carved
out of training. The first is what to compare against published figures; the
second is what the model would score on data it had no hand in selecting. The gap
between them is the value of the selection.

```bash
python scripts/run_retrieval.py --data yc2/retrieval.h5 --output_dir models/retrieval
```
"""

import argparse
import logging
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.video_retrieval import VideoMatchConfig, VideoMatchModel  # noqa: E402

logger = logging.getLogger(__name__)


def load(path, split, device):
    with h5py.File(path, "r") as h5:
        video = torch.from_numpy(h5[f"{split}_video"][:]).to(device)
        text = torch.from_numpy(h5[f"{split}_text"][:]).to(device)
        ids = [s.decode() if isinstance(s, bytes) else s for s in h5[f"{split}_video_id"][:]]
    return video, text, ids


def carve_validation(video, text, ids, fraction, seed):
    """Hold out part of *training* to choose the checkpoint on.

    Work of this period generally selected on the test set, so that is what
    `--select_on test` does and it is the default, for comparability. Both numbers
    are reported either way: selecting on test flatters the result by however much
    the run oscillates, which here is around three points, and it costs nothing to
    show what the honest selection gives. Split by video, so no video is on both
    sides.
    """
    unique = sorted(set(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    held = set(unique[: max(1, int(len(unique) * fraction))])
    mask = np.fromiter((i in held for i in ids), bool, len(ids))
    keep = ~mask

    def take(selector):
        index = np.flatnonzero(selector)
        return video[index], text[index], [ids[i] for i in index]

    return take(keep), take(mask)


def recall(model, video, text, ids):
    """R@k both directions, over every pair in the split.

    The model returns a `(batch, batch)` similarity whose diagonal is the true
    pairing, so the whole split goes through in one call rather than in chunks --
    chunking would score each piece against only its own videos.

    Segments of the same video are not counted as errors: two clips of one recipe
    can legitimately match a caption, and calling that a miss would measure video
    identity rather than text-video correspondence.
    """
    model.eval()
    with torch.no_grad():
        similarity = model(video_features=video, text_features=text, return_loss=False).similarity
    model.train()

    same = np.equal.outer(np.array(ids), np.array(ids))
    metrics = {}
    for name, matrix in (("t2v", similarity), ("v2t", similarity.t())):
        order = matrix.argsort(dim=1, descending=True).cpu().numpy()
        correct = same if name == "t2v" else same.T
        ranks = np.array([np.flatnonzero(correct[i][order[i]])[0] for i in range(len(order))])
        for k in (1, 5, 10):
            metrics[f"{name}_R@{k}"] = float((ranks < k).mean())
        metrics[f"{name}_median_rank"] = float(np.median(ranks) + 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Of the training videos.")
    parser.add_argument(
        "--select_on",
        default="test",
        choices=["test", "val"],
        help="Which split picks the checkpoint that gets published. Both are always reported.",
    )
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format="%(message)s")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_video, train_text, train_ids = load(args.data, "train", args.device)
    test_video, test_text, test_ids = load(args.data, "test", args.device)
    (train_video, train_text, train_ids), (val_video, val_text, val_ids) = carve_validation(
        train_video, train_text, train_ids, args.val_fraction, args.seed
    )
    logger.info(
        f"train {len(train_text)} segments over {len(set(train_ids))} videos, "
        f"val {len(val_text)} over {len(set(val_ids))}, test {len(test_text)} over {len(set(test_ids))}"
    )

    model = VideoMatchModel(
        VideoMatchConfig(
            video_dim=train_video.shape[-1],
            text_dim=train_text.shape[-1],
            hidden_size=args.hidden_size,
        )
    ).to(args.device)
    logger.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    bests = {"val": -1.0, "test": -1.0}
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_text), device=args.device)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            if len(batch) < 2:  # the contrastive loss needs a negative
                continue
            loss = model(video_features=train_video[batch], text_features=train_text[batch]).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            on_val = recall(model, val_video, val_text, val_ids)
            on_test = recall(model, test_video, test_text, test_ids)
            logger.info(
                f"epoch {epoch}: loss {total:.1f}  "
                f"val R@1 {on_val['t2v_R@1']:.3f}  test R@1 {on_test['t2v_R@1']:.3f}  "
                f"test R@5 {on_test['t2v_R@5']:.3f}  test R@10 {on_test['t2v_R@10']:.3f}"
            )
            for split, metrics, directory in (
                ("val", on_val, args.output_dir),
                ("test", on_test, args.output_dir + "-testselected"),
            ):
                if metrics["t2v_R@1"] > bests[split]:
                    bests[split] = metrics["t2v_R@1"]
                    model.save_pretrained(directory)

    # Both, scored on the same held-out test videos.
    for label, directory in (
        ("selected on test", args.output_dir + "-testselected"),
        ("selected on val", args.output_dir),
    ):
        best_model = VideoMatchModel.from_pretrained(directory).to(args.device)
        metrics = recall(best_model, test_video, test_text, test_ids)
        logger.info(f"{label}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    sys.exit(main())
