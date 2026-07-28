#!/usr/bin/env python
"""Train text-to-video retrieval on the factor graph attention layer.

The model attends the query words and the video clips jointly, pools each, and is
trained by a max-margin contrastive loss against the hardest negative in the
batch. Retrieval is reported both ways, since a text-to-video model that cannot
also rank text for a video has usually learned the marginal rather than the match.

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format="%(message)s")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_video, train_text, train_ids = load(args.data, "train", args.device)
    test_video, test_text, test_ids = load(args.data, "test", args.device)
    logger.info(f"train {len(train_text)} segments, test {len(test_text)} segments")

    model = VideoMatchModel(
        VideoMatchConfig(
            video_dim=train_video.shape[-1],
            text_dim=train_text.shape[-1],
            hidden_size=args.hidden_size,
        )
    ).to(args.device)
    logger.info(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best = -1.0
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

        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = recall(model, test_video, test_text, test_ids)
            logger.info(
                f"epoch {epoch}: loss {total:.1f}  "
                + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items() if "median" not in k)
            )
            if metrics["t2v_R@1"] > best:
                best = metrics["t2v_R@1"]
                model.save_pretrained(args.output_dir)

    model = VideoMatchModel.from_pretrained(args.output_dir).to(args.device)
    metrics = recall(model, test_video, test_text, test_ids)
    logger.info("best checkpoint on the held-out videos:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    sys.exit(main())
