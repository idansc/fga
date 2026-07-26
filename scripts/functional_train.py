#!/usr/bin/env python
"""Functional training runs for the tasks whose datasets are not obtainable.

The AVSD features were lost with the same expired SharePoint as VisDial's, and
VideoMatch's DiDeMo/ActivityNet features were never publicly posted — so those
two cannot be trained on their real data without a feature-regeneration project
of their own. This script is the strongest check available short of that: each
task trains on synthetic data with *planted structure*, and passes only if the
model recovers that structure to a target metric on held-out examples.

That is a much stronger claim than a smoke test (loss decreased for 30 steps),
and a much weaker one than a benchmark number. It certifies the wiring — the
attention finds signal that is only visible through the right modality
interactions — not task accuracy.

    retrieval:  each text is a noisy projection of its video's latent; the model
                must rank the true video first among 500 held-out candidates.
    avsd:       exactly one of the four video streams correlates with the
                question; a linear probe must say which. The probe reads the
                *temporal* state -- the LSTM-fused streams -- because that is
                where cross-stream information lives; the decoder state carries
                the question and history, not the stream comparison.
    navigation: every cell carries its quadrant's code and the target marks one
                cell; REINFORCE must act toward that quadrant. The quadrant code
                has to survive pooling, since attention reduces the grid to one
                vector -- an earlier version planted only a positional vector per
                cell and had an oracle ceiling of 0.63, which certified nothing.

```bash
python scripts/functional_train.py --task retrieval
```
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_retrieval(steps: int = 1500) -> bool:
    """Text must retrieve its own video among 500 held-out distractors."""
    from fga.tasks.video_retrieval import VideoMatchConfig, VideoMatchModel

    torch.manual_seed(0)
    dim, clips, words = 256, 12, 8

    def sample(n):
        latent = torch.randn(n, dim, device=DEVICE)
        video = latent[:, None, :] + 0.6 * torch.randn(n, clips, dim, device=DEVICE)
        text = latent[:, None, :] @ projection + 0.6 * torch.randn(n, words, dim, device=DEVICE)
        return video, text

    projection = torch.randn(dim, dim, device=DEVICE) / dim**0.5
    model = VideoMatchModel(VideoMatchConfig(video_dim=dim, text_dim=dim, hidden_size=256)).to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for step in range(steps):
        video, text = sample(64)
        loss = model(video_features=video, text_features=text).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print(f"  step {step}: loss {float(loss):.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        video, text = sample(500)
        out = model(video_features=video, text_features=text, return_loss=False)
        ranks = out.similarity.argsort(dim=1, descending=True)
        r1 = (ranks[:, 0] == torch.arange(500, device=DEVICE)).float().mean().item()
        r10 = (ranks[:, :10] == torch.arange(500, device=DEVICE)[:, None]).any(1).float().mean().item()
    print(f"retrieval held-out: R@1 {r1:.3f}  R@10 {r10:.3f}  (chance 0.002 / 0.02)")
    return r1 > 0.5


def run_avsd(steps: int = 1500) -> bool:
    """A probe on the fused state must identify which stream matches the question."""
    from fga.tasks.video_dialog import AVSDConfig, AVSDEncoder

    torch.manual_seed(0)
    config = AVSDConfig(
        question_dim=128,
        video_dim=128,
        audio_dim=64,
        hidden_size=128,
        num_video_regions=25,
        max_question_length=8,
        num_audio_steps=6,
        history_dim=64,
    )
    encoder = AVSDEncoder(config).to(DEVICE).train()
    probe = torch.nn.Linear(config.hidden_size, 4).to(DEVICE)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(probe.parameters()), lr=3e-4)

    def sample(n):
        cue = torch.randn(n, 128, device=DEVICE)
        question = cue[:, None, :] + 0.5 * torch.randn(n, 8, 128, device=DEVICE)
        video = torch.randn(n, 4, 25, 128, device=DEVICE)
        which = torch.randint(0, 4, (n,), device=DEVICE)
        video[torch.arange(n), which] += cue[:, None, :]
        audio = torch.randn(n, 6, 64, device=DEVICE)
        history = torch.randn(n, 64, device=DEVICE)
        return question, video, audio, history, which

    for step in range(steps):
        question, video, audio, history, which = sample(64)
        fused = encoder(
            question_states=question, video_features=video, audio_features=audio, history_state=history
        ).temporal_state
        loss = F.cross_entropy(probe(fused), which)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print(f"  step {step}: loss {float(loss):.3f}", flush=True)

    encoder.eval()
    with torch.no_grad():
        question, video, audio, history, which = sample(2000)
        fused = encoder(
            question_states=question, video_features=video, audio_features=audio, history_state=history
        ).temporal_state
        accuracy = (probe(fused).argmax(-1) == which).float().mean().item()
    print(f"avsd held-out: which-stream accuracy {accuracy:.3f}  (chance 0.25)")
    return accuracy > 0.9


def run_navigation(steps: int = 4000) -> bool:
    """REINFORCE: act toward the quadrant of the grid holding the target."""
    from fga.tasks.navigation import NavigationConfig, NavigationPolicy

    torch.manual_seed(0)
    grid_side, dim = 6, 128
    config = NavigationConfig(
        target_dim=dim, observation_dim=dim, grid_size=grid_side * grid_side, hidden_size=128, action_space=4
    )
    policy = NavigationPolicy(config).to(DEVICE).train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Each cell carries the code of its quadrant, so the answer survives the
    # pooling that attention performs. Real observation features carry this kind
    # of spatial structure; independent random vectors do not.
    quadrant_code = torch.randn(4, dim, device=DEVICE) * 3.0
    cells = torch.arange(grid_side * grid_side, device=DEVICE)
    cell_quadrant = (cells // grid_side // (grid_side // 2)) * 2 + (cells % grid_side) // (grid_side // 2)

    def sample(n):
        target = torch.randn(n, 1, dim, device=DEVICE)
        cell = torch.randint(0, grid_side * grid_side, (n,), device=DEVICE)
        observation = 0.5 * torch.randn(n, grid_side * grid_side, dim, device=DEVICE) + quadrant_code[cell_quadrant]
        observation[torch.arange(n), cell] += 3.0 * target[:, 0]
        return target, observation, cell_quadrant[cell]

    success = 0.0
    for step in range(steps):
        target, observation, quadrant = sample(128)
        out = policy(target_embeds=target, observation=observation)
        distribution = torch.distributions.Categorical(logits=out.action_logits)
        action = distribution.sample()
        reward = (action == quadrant).float()
        # REINFORCE with the critic as baseline, plus the value loss.
        advantage = reward - out.value.detach()
        loss = -(distribution.log_prob(action) * advantage).mean() + F.mse_loss(out.value, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        success = 0.98 * success + 0.02 * reward.mean().item()
        if step % 500 == 0:
            print(f"  step {step}: running success {success:.3f}", flush=True)

    policy.eval()
    with torch.no_grad():
        target, observation, quadrant = sample(2000)
        action = policy(target_embeds=target, observation=observation).action_logits.argmax(-1)
        accuracy = (action == quadrant).float().mean().item()
    print(f"navigation held-out: quadrant accuracy {accuracy:.3f}  (chance 0.25)")
    return accuracy > 0.9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["retrieval", "avsd", "navigation", "all"], default="all")
    args = parser.parse_args()

    runs = {"retrieval": run_retrieval, "avsd": run_avsd, "navigation": run_navigation}
    selected = runs if args.task == "all" else {args.task: runs[args.task]}

    results = {}
    for name, fn in selected.items():
        print(f"=== {name} ===", flush=True)
        results[name] = fn()

    print()
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
