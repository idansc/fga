#!/usr/bin/env python
"""Score the original navigation model's own weights in this environment.

The published 46.2 success / 17.9 SPL comes with released checkpoints. Running
those weights here separates two questions that are otherwise tangled: is this
offline environment faithful, and is the gap in how the policy is trained?

* Their checkpoint scores near 46 here -> the environment and the episode set are
  right, and the difference is training.
* It scores well below -> something in this environment differs from theirs, and
  no amount of training would have closed the gap.

The forward pass below is `BaseModel.embedding` from barmayo/spatial_attention,
rewritten to take a batch. The original squeezes to a single episode throughout —
`theta_ST` comes out as `(49,)` and the softmax runs over `dim=0` — so it only
ever ran one episode at a time.

```bash
python scripts/eval_original_navigation.py \
    --checkpoint nav/EOTP_final_89329011_6000000_2020-10-09_16:54:35.dat \
    --data_root nav/data --test_episodes nav/data/test_episodes.json
```
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fga.tasks.navigation.environment import (  # noqa: E402
    ACTIONS,
    GloveTargets,
    load_scenes,
    load_test_episodes,
)


class Chomp1d(nn.Module):
    """Trim the right padding, which is what makes the convolution causal."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, : -self.size].contiguous()


class TemporalConvNet(nn.Module):
    """The learned-loss network, `ll_tc`.

    Two causal blocks. Note that each block defines a second convolution which its
    forward never applies -- the weights are in the checkpoint and are dead. This
    reproduces that rather than fixing it, because the released weights were
    trained with it.

    The steps of the interaction history are the *channels* and the feature axis
    is the sequence, so the network mixes across time at every feature position.
    """

    def __init__(self, num_steps, channels=(10, 1), kernel_size=2):
        super().__init__()
        self.num_levels = len(channels)
        for i, out_channels in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            in_channels = num_steps if i == 0 else channels[i - 1]
            block = nn.Module()
            block.ll_conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation
            )
            block.ll_conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation
            )
            block.chomp1 = Chomp1d(padding)
            block.dilation, block.padding = dilation, padding
            setattr(self, f"ll_temporal_block{i}", block)

    def forward(self, x, params=None):
        for i in range(self.num_levels):
            block = getattr(self, f"ll_temporal_block{i}")
            if params is None:
                x = block.ll_conv1(x)
            else:
                x = F.conv1d(
                    x,
                    weight=params[f"ll_tc.ll_temporal_block{i}.ll_conv1.weight"],
                    bias=params[f"ll_tc.ll_temporal_block{i}.ll_conv1.bias"],
                    stride=1,
                    padding=block.padding,
                    dilation=block.dilation,
                )
            x = block.chomp1(x)
            x = F.leaky_relu(x)
        return x


class OriginalPolicy(nn.Module):
    """barmayo/spatial_attention's BaseModel, batched.

    Every grid cell is scored by cosine similarity against the target, the
    recurrent memory and the last action; the three maps are mixed by weights
    `memory_alpha` predicts from the memory, and the softmaxed result scales the
    grid, which is flattened into the recurrence.
    """

    def __init__(self, glove_dim=300, hidden=512, actions=6, num_steps=6):
        super().__init__()
        self.embed_state = nn.Conv1d(hidden, 64, 1)
        self.embed_glove = nn.Conv1d(glove_dim, 64, 1)
        self.embed_action = nn.Conv1d(actions, 64, 1)
        self.embed_memory = nn.Conv1d(hidden, 64, 1)
        self.memory_alpha = nn.Conv1d(hidden, 3, 1)
        self.lstm = nn.LSTMCell(3136, hidden)
        self.critic_linear = nn.Linear(hidden, 1)
        self.actor_linear = nn.Linear(hidden, actions)
        self.ll_tc = TemporalConvNet(num_steps)
        self.num_steps = num_steps

    def forward(self, state, target, prev_action, hidden, params=None):
        """`params` runs the policy on adapted weights instead of its own."""

        def conv(name, x):
            if params is None:
                return getattr(self, name)(x)
            return F.conv1d(x, params[f"{name}.weight"], params[f"{name}.bias"])

        memory, cell = hidden
        state = state.transpose(1, 2)                       # (B, 512, 49)
        state_embedding = conv("embed_state", state)        # (B, 64, 49)

        state_att = F.normalize(state_embedding, dim=1)
        glove_att = F.normalize(conv("embed_glove", target.unsqueeze(2)), dim=1)
        action_att = F.normalize(conv("embed_action", prev_action.unsqueeze(2)), dim=1)
        memory_att = F.normalize(conv("embed_memory", memory.unsqueeze(2)), dim=1)

        cells = state_att.transpose(1, 2)                   # (B, 49, 64)
        theta = torch.cat(
            [cells.bmm(glove_att), cells.bmm(memory_att), cells.bmm(action_att)], dim=2
        )                                                   # (B, 49, 3)
        alpha = conv("memory_alpha", memory.unsqueeze(2))   # (B, 3, 1)
        poten = F.softmax(theta.bmm(alpha).squeeze(2), dim=1)

        fused = (poten.unsqueeze(1) * F.relu(state_embedding)).reshape(state.size(0), -1)
        if params is None:
            memory, cell = self.lstm(fused, (memory, cell))
            return self.actor_linear(memory), self.critic_linear(memory).squeeze(-1), (memory, cell)
        memory, cell = torch._VF.lstm_cell(
            fused, (memory, cell),
            params["lstm.weight_ih"], params["lstm.weight_hh"],
            params["lstm.bias_ih"], params["lstm.bias_hh"],
        )
        logits = F.linear(memory, params["actor_linear.weight"], params["actor_linear.bias"])
        value = F.linear(memory, params["critic_linear.weight"], params["critic_linear.bias"])
        return logits, value.squeeze(-1), (memory, cell)

    def learned_loss(self, history, params):
        """A loss computed from the episode so far, with no reward in it.

        The agent cannot know whether it is succeeding mid-episode, so this
        network -- trained end to end to be useful when descended -- stands in for
        the signal. Its value is the norm of what the temporal convolution makes
        of the last few interactions.
        """
        return self.ll_tc(history.unsqueeze(0), params).squeeze(0).pow(2).sum(1).pow(0.5)


def run_adapted(policy, episodes, targets, args):
    """Each episode adapts the policy to itself, then keeps acting with it.

    Every `num_steps` interactions the learned loss is computed over that window
    and one gradient step is taken, giving weights specific to this episode. No
    reward is used, so this is exactly what an agent could do at test time.
    """
    base = {name: tensor for name, tensor in policy.named_parameters()}
    adaptable = [k for k in base if not k.startswith("ll_tc")]
    successes, spl = [], []

    for index, episode in enumerate(episodes):
        episode.steps, episode.done, episode.success = 0, False, False
        episode.state = episode.start_state
        optimal = episode.optimal_steps()

        params = {k: v.clone() for k, v in base.items()}
        target = torch.from_numpy(targets[episode.target]).to(args.device).unsqueeze(0)
        memory = torch.zeros(1, 512, device=args.device)
        hidden = (memory, memory.clone())
        prev_action = torch.zeros(1, len(ACTIONS), device=args.device)
        history = []

        while not episode.done:
            observation = torch.from_numpy(episode.observation()).to(args.device).unsqueeze(0)
            logits, _, hidden = policy(observation, target, prev_action, hidden, params)
            probabilities = F.softmax(logits, dim=1)
            # The interaction the learned loss reads: what the agent knew and what
            # it was about to do.
            history.append(torch.cat((hidden[0], probabilities), dim=1))

            action = logits.argmax(-1)
            episode.step(int(action))
            prev_action = F.one_hot(action, len(ACTIONS)).float()

            if len(history) % args.num_steps == 0 and not episode.done:
                loss = policy.learned_loss(torch.cat(history[-args.num_steps :], dim=0), params).sum()
                grads = torch.autograd.grad(
                    loss, [params[k] for k in adaptable], allow_unused=True, retain_graph=False
                )
                params = dict(params)
                for key, grad in zip(adaptable, grads):
                    if grad is not None:
                        params[key] = params[key] - args.inner_lr * grad
                # The window is closed; nothing after it should backpropagate into it.
                hidden = (hidden[0].detach(), hidden[1].detach())
                params = {k: v.detach().requires_grad_(True) for k, v in params.items()}
                history = []

        successes.append(float(episode.success))
        spl.append(optimal / max(episode.steps, optimal) if episode.success and optimal else 0.0)
        if index % 400 == 0:
            print(f"  {index}/{len(episodes)}  running success {np.mean(successes):.3f}", flush=True)

    return successes, spl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--test_episodes", required=True)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Adapt the policy inside each episode by descending the learned loss, which is what "
        "the published number does. Runs one episode at a time, since the weights become per-episode.",
    )
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--num_steps", type=int, default=6, help="Interactions per adaptation step.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not any(hasattr(v, "shape") for v in state.values()):
        state = next(v for v in state.values() if isinstance(v, dict))

    policy = OriginalPolicy(num_steps=args.num_steps).to(args.device).eval()
    loaded = policy.load_state_dict({k: v for k, v in state.items() if k in policy.state_dict()}, strict=False)
    print(f"loaded {len(policy.state_dict()) - len(loaded.missing_keys)} tensors; missing {loaded.missing_keys}")
    print(f"ignored (the learned-loss head, unused at evaluation): "
          f"{sorted(k for k in state if k.startswith('ll_tc'))[:2]} ...")

    scenes = {s.name: s for s in load_scenes(os.path.join(args.data_root, "thor_offline_data"), split="test")}
    targets = GloveTargets(os.path.join(args.data_root, "thor_glove", "glove_map300d.hdf5"))
    episodes = load_test_episodes(args.test_episodes, scenes, args.max_steps)
    print(f"{len(episodes)} published episodes over {len({e.scene.name for e in episodes})} scenes")

    if args.adapt:
        successes, spl = run_adapted(policy, episodes, targets, args)
        print(f"\noriginal weights, adapting in-episode: success {np.mean(successes):.3f}  SPL {np.mean(spl):.3f}")
        print("published: success 0.462  SPL 0.179")
        return

    successes, spl = [], []
    with torch.no_grad():
        for start in range(0, len(episodes), args.batch):
            group = episodes[start : start + args.batch]
            for episode in group:
                episode.steps, episode.done, episode.success = 0, False, False
                episode.state = episode.start_state
            optimal = [e.optimal_steps() for e in group]

            live = list(range(len(group)))
            memory = torch.zeros(len(group), 512, device=args.device)
            hidden = (memory, memory.clone())
            prev_action = torch.zeros(len(group), len(ACTIONS), device=args.device)

            while live:
                active = [group[i] for i in live]
                observation = torch.from_numpy(np.stack([e.observation() for e in active])).to(args.device)
                target = torch.from_numpy(np.stack([targets[e.target] for e in active])).to(args.device)
                logits, _, hidden = policy(observation, target, prev_action, hidden)
                action = logits.argmax(-1)

                keep = []
                for slot, index in enumerate(live):
                    group[index].step(int(action[slot]))
                    if not group[index].done:
                        keep.append(slot)
                hidden = (hidden[0][keep], hidden[1][keep])
                prev_action = F.one_hot(action[keep], len(ACTIONS)).float()
                live = [live[slot] for slot in keep]

            for episode, best in zip(group, optimal):
                successes.append(float(episode.success))
                spl.append(best / max(episode.steps, best) if episode.success and best else 0.0)
            if start % (args.batch * 10) == 0:
                print(f"  {start}/{len(episodes)}  running success {np.mean(successes):.3f}", flush=True)

    print(f"\noriginal weights in this environment: success {np.mean(successes):.3f}  SPL {np.mean(spl):.3f}")
    print("published: success 0.462  SPL 0.179")


if __name__ == "__main__":
    sys.exit(main())
