#!/usr/bin/env python
"""Train target-driven navigation on the offline AI2-THOR episodes.

Unlike the other tasks here there is no dataset to iterate: the agent generates
its own data by acting, and is trained by advantage actor-critic on the returns.
Episodes are replayed from cached ResNet18 features, so this needs no simulator.

Many episodes run in lockstep, each resetting to a fresh scene and target as it
ends. That is the usual A2C arrangement, and it is required rather than merely
faster here: the attention batch-normalizes its interaction grid, which is
undefined on a batch of one.

Reports the two standard measures on held-out scenes:

    success    fraction of episodes that end with `Done` while the target is visible
    SPL        success weighted by how close the route was to the shortest one,
               `mean(success * optimal / max(taken, optimal))`, so an agent that
               finds the target by exhausting the room scores far below one that
               walks to it

```bash
python scripts/run_navigation.py --data_root nav/data --output_dir models/navigation \
    --num_episodes 200000
```
"""

import argparse
import logging
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.navigation import NavigationConfig, NavigationPolicy  # noqa: E402
from fga.tasks.navigation.environment import (  # noqa: E402
    ACTIONS,
    GloveTargets,
    NavigationEpisode,
    load_scenes,
)

logger = logging.getLogger(__name__)


def stack_observations(episodes, device):
    return torch.from_numpy(np.stack([e.observation() for e in episodes])).to(device)


def stack_targets(episodes, targets, device):
    return torch.from_numpy(np.stack([targets[e.target] for e in episodes])).to(device).unsqueeze(1)


def evaluate(policy, scenes, targets, device, episodes, max_steps, seed=0, batch=32):
    """Success and SPL over fresh episodes, acting greedily.

    Batched only for speed -- the attention uses its running batch-norm
    statistics in eval mode, so the result does not depend on the batch.
    """
    rng = random.Random(seed)
    policy.eval()
    successes, spl = [], []
    with torch.no_grad():
        remaining = episodes
        while remaining > 0:
            group = [NavigationEpisode.sample(scenes, rng, max_steps) for _ in range(min(batch, remaining))]
            remaining -= len(group)
            optimal = [e.optimal_steps() for e in group]
            live = list(range(len(group)))
            hidden = None
            prev_action = torch.zeros(len(group), len(ACTIONS), device=device)

            while live:
                active = [group[i] for i in live]
                out = policy(
                    target_embeds=stack_targets(active, targets, device),
                    observation=stack_observations(active, device),
                    hidden_state=hidden,
                    prev_action=prev_action,
                )
                actions = out.action_logits.argmax(-1)
                keep = []
                for slot, index in enumerate(live):
                    group[index].step(int(actions[slot]))
                    if not group[index].done:
                        keep.append(slot)
                hidden = (out.hidden_state[0][keep], out.hidden_state[1][keep])
                prev_action = F.one_hot(actions[keep], len(ACTIONS)).float()
                live = [live[slot] for slot in keep]

            for episode, best in zip(group, optimal):
                successes.append(float(episode.success))
                spl.append(best / max(episode.steps, best) if episode.success and best else 0.0)
    policy.train()
    return float(np.mean(successes)), float(np.mean(spl))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="Directory holding thor_offline_data and thor_glove.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=200000)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--num_envs", type=int, default=32, help="Episodes stepped in lockstep.")
    parser.add_argument("--rollout_steps", type=int, default=30, help="Steps per update.")
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0, help="On the fused state, against scene overfitting.")
    parser.add_argument("--value_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=50.0)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format="%(message)s")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    offline = os.path.join(args.data_root, "thor_offline_data")
    targets = GloveTargets(os.path.join(args.data_root, "thor_glove", "glove_map300d.hdf5"))
    train_scenes = load_scenes(offline, split="train")
    test_scenes = load_scenes(offline, split="test")
    logger.info(f"{len(train_scenes)} train scenes, {len(test_scenes)} test scenes, glove dim {targets.dim}")

    sample = train_scenes[0].feature(next(iter(train_scenes[0].states)))
    policy = NavigationPolicy(
        NavigationConfig(
            target_dim=targets.dim,
            observation_dim=sample.shape[1],
            grid_size=sample.shape[0],
            action_space=len(ACTIONS),
            dropout=args.dropout,
        )
    ).to(args.device)
    logger.info(f"params: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    recent = deque(maxlen=1000)
    best_spl = -1.0
    episodes_done = 0
    next_eval = args.eval_every

    # Episodes run in lockstep and each is replaced the moment it ends, so the
    # batch is always full and the batch-norm always well defined.
    envs = [NavigationEpisode.sample(train_scenes, rng, args.max_steps) for _ in range(args.num_envs)]
    hidden = None
    prev_action = torch.zeros(args.num_envs, len(ACTIONS), device=args.device)

    while episodes_done < args.num_episodes:
        log_probs, values, entropies, rewards, masks = [], [], [], [], []

        for _ in range(args.rollout_steps):
            out = policy(
                target_embeds=stack_targets(envs, targets, args.device),
                observation=stack_observations(envs, args.device),
                hidden_state=hidden,
                prev_action=prev_action,
            )
            distribution = torch.distributions.Categorical(logits=out.action_logits)
            action = distribution.sample()

            step_rewards, alive = [], []
            for slot, episode in enumerate(envs):
                reward, finished, _ = episode.step(int(action[slot]))
                step_rewards.append(reward)
                alive.append(0.0 if finished else 1.0)
                if finished:
                    recent.append(float(episode.success))
                    episodes_done += 1
                    envs[slot] = NavigationEpisode.sample(train_scenes, rng, args.max_steps)

            log_probs.append(distribution.log_prob(action))
            values.append(out.value)
            entropies.append(distribution.entropy())
            rewards.append(torch.tensor(step_rewards, device=args.device))
            mask = torch.tensor(alive, device=args.device)
            masks.append(mask)

            # A finished slot now holds a fresh episode, so its memory and last
            # action must not carry over from the episode that just ended.
            hidden = (out.hidden_state[0] * mask.unsqueeze(1), out.hidden_state[1] * mask.unsqueeze(1))
            prev_action = F.one_hot(action, len(ACTIONS)).float() * mask.unsqueeze(1)

        # Bootstrap the tail of each unfinished episode with its own value.
        with torch.no_grad():
            bootstrap = policy(
                target_embeds=stack_targets(envs, targets, args.device),
                observation=stack_observations(envs, args.device),
                hidden_state=hidden,
                prev_action=prev_action,
            ).value

        returns, running = [], bootstrap
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running = reward + args.gamma * running * mask
            returns.append(running)
        returns = torch.stack(returns[::-1])

        values = torch.stack(values)
        advantage = returns - values.detach()
        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + args.value_weight * value_loss - args.entropy_weight * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()
        # The recurrence continues into the next rollout, but its graph does not.
        hidden = (hidden[0].detach(), hidden[1].detach())
        prev_action = prev_action.detach()

        if episodes_done >= next_eval:
            next_eval += args.eval_every
            success, spl = evaluate(
                policy, test_scenes, targets, args.device, args.eval_episodes, args.max_steps
            )
            logger.info(
                f"{episodes_done} episodes: train success {np.mean(recent) if recent else 0:.3f}  "
                f"HELD-OUT success {success:.3f}  SPL {spl:.3f}"
            )
            if spl > best_spl:
                best_spl = spl
                policy.save_pretrained(args.output_dir)
                logger.info(f"  saved to {args.output_dir}")

    success, spl = evaluate(policy, test_scenes, targets, args.device, 1000, args.max_steps)
    logger.info(f"final held-out over 1000 episodes: success {success:.3f}  SPL {spl:.3f}")


if __name__ == "__main__":
    main()
