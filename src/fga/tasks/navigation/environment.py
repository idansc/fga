"""Offline AI2-THOR episodes, replayed from cached features.

Target-driven navigation normally needs the AI2-THOR simulator in the loop. It
does not have to: every reachable pose in a scene was rendered once and its
ResNet18 feature map cached, so an episode is a walk over a graph of poses with a
tensor lookup at each node. That is what the SAVN release distributes and what
this module reads, so the task trains with no simulator, no GPU renderer and no
Unity.

A **state** is a pose, written `"x|z|rotation|horizon"` — position on a 0.25 m
grid, heading in 45 degree steps, camera pitch in 30 degree steps. The agent's
six actions move between poses:

    MoveAhead   one grid step along the current heading (8-way)
    RotateLeft / RotateRight    heading -+ 45 degrees
    LookUp / LookDown           pitch -+ 30 degrees, within range
    Done        claim the target is visible; ends the episode either way

An action that would leave the scene's graph fails and the pose is unchanged,
which is the signal the agent has hit a wall.

Expects the SAVN offline dump:

    data/thor_offline_data/FloorPlan<N>_physics/
        resnet18_featuremap.hdf5    pose -> (1, 512, 7, 7)
        graph.json                  node-link graph of reachable poses
        visible_object_map.json     object instance -> poses it is visible from
    data/thor_glove/glove_map300d.hdf5   object class -> 300-d embedding

```bash
curl -O https://prior-datasets.s3.us-east-2.amazonaws.com/savn/data.tar.gz
```
"""

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

__all__ = [
    "ACTIONS",
    "DONE",
    "OfflineScene",
    "GloveTargets",
    "NavigationEpisode",
    "scene_names",
    "ROOM_OBJECTS",
]

#: Action order. `Done` is last, matching the original's `DONE_ACTION_INT = 5`.
ACTIONS = ("MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Done")
DONE = "Done"

GRID_SIZE = 0.25
GOAL_SUCCESS_REWARD = 5.0
STEP_PENALTY = -0.01

#: Which object classes are searched for in each room type, from the original's
#: datasets/constants.py. A target only counts if the scene actually contains it.
ROOM_OBJECTS = {
    "kitchen": ["Toaster", "Microwave", "Fridge", "CoffeeMaker", "GarbageCan", "Box", "Bowl"],
    "living_room": ["Pillow", "Laptop", "Television", "GarbageCan", "Box", "Bowl"],
    "bedroom": ["HousePlant", "Lamp", "Book", "AlarmClock"],
    "bathroom": ["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
}

#: AI2-THOR numbers its scenes by room type: 1-30 kitchens, 201-230 living rooms,
#: 301-330 bedrooms, 401-430 bathrooms, with 1-20 of each held for training.
_ROOM_OFFSETS = {"kitchen": 0, "living_room": 200, "bedroom": 300, "bathroom": 400}


def scene_names(rooms: Sequence[str] = tuple(ROOM_OBJECTS), split: str = "train") -> List[str]:
    """Scene folder names for a split: 1-20 of each room type train, 21-30 test."""
    numbers = range(1, 21) if split == "train" else range(21, 31)
    return [f"FloorPlan{_ROOM_OFFSETS[room] + n}_physics" for room in rooms for n in numbers]


def room_of(scene: str) -> str:
    number = int(scene.replace("FloorPlan", "").replace("_physics", ""))
    for room, offset in sorted(_ROOM_OFFSETS.items(), key=lambda kv: -kv[1]):
        if number > offset:
            return room
    return "kitchen"


class GloveTargets:
    """Object class -> embedding, the vector that tells the agent what to find."""

    def __init__(self, path: str):
        with h5py.File(path, "r") as handle:
            self.vectors = {k: np.asarray(v, dtype=np.float32) for k, v in handle.items()}

    def __contains__(self, name: str) -> bool:
        return name in self.vectors

    def __getitem__(self, name: str) -> np.ndarray:
        return self.vectors[name]

    @property
    def dim(self) -> int:
        return len(next(iter(self.vectors.values())))


class OfflineScene:
    """One scene: the poses, what connects them, and what is visible from each.

    Args:
        directory: a `FloorPlan<N>_physics` folder from the SAVN dump.
        features_file: which cached feature map to read.
        keep_features_open: hold the HDF5 handle open. Handles do not survive a
            fork, so a dataloader worker reopens lazily.
    """

    def __init__(
        self,
        directory: str,
        features_file: str = "resnet18_featuremap.hdf5",
        keep_features_open: bool = True,
    ):
        self.directory = directory
        self.name = os.path.basename(directory)
        self.features_path = os.path.join(directory, features_file)
        self._features: Optional[h5py.File] = None

        with open(os.path.join(directory, "graph.json")) as handle:
            graph = json.load(handle)
        self.states = {node["id"] for node in graph["nodes"]}
        # MoveAhead is only legal along an edge; every other action is a pure
        # rotation or pitch change and stays at the same position.
        self.moves: Dict[str, set] = {state: set() for state in self.states}
        for link in graph["links"]:
            self.moves[link["source"]].add(link["target"])

        with open(os.path.join(directory, "visible_object_map.json")) as handle:
            visible = json.load(handle)
        # Keys are instances ("Fridge|-01.5|+00.9|+02.3"); the agent is asked for
        # a class, and any instance of it will do.
        self.goals: Dict[str, set] = {}
        for instance, poses in visible.items():
            self.goals.setdefault(instance.split("|")[0], set()).update(poses)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def targets(self) -> List[str]:
        """Object classes this scene can pose as a target, in its room type."""
        return [name for name in ROOM_OBJECTS[room_of(self.name)] if self.goals.get(name)]

    def feature(self, state: str) -> np.ndarray:
        """The cached observation at a pose, as `(49, 512)` — cells, then channels."""
        if self._features is None:
            self._features = h5py.File(self.features_path, "r", swmr=True)
        grid = np.asarray(self._features[state], dtype=np.float32)  # (1, 512, 7, 7)
        return grid.reshape(grid.shape[-3], -1).T

    def next_state(self, state: str, action: str) -> Optional[str]:
        """The pose an action leads to, or `None` if it is not possible here."""
        x, z, rotation, horizon = state.split("|")
        rotation, horizon = int(rotation), int(horizon)

        if action == "MoveAhead":
            dx, dz = {
                0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1),
                180: (0, -1), 225: (-1, -1), 270: (-1, 0), 315: (-1, 1),
            }[rotation]
            candidate = f"{float(x) + dx * GRID_SIZE:.2f}|{float(z) + dz * GRID_SIZE:.2f}|{rotation}|{horizon}"
            # Walls and furniture are encoded as missing edges, not as geometry.
            return candidate if candidate in self.moves[state] else None
        if action == "RotateLeft":
            candidate = f"{x}|{z}|{(rotation - 45) % 360}|{horizon}"
        elif action == "RotateRight":
            candidate = f"{x}|{z}|{(rotation + 45) % 360}|{horizon}"
        elif action == "LookUp":
            if horizon <= 0:
                return None
            candidate = f"{x}|{z}|{rotation}|{horizon - 30}"
        elif action == "LookDown":
            if horizon >= 30:
                return None
            candidate = f"{x}|{z}|{rotation}|{horizon + 30}"
        else:
            raise ValueError(f"Unknown action {action!r}.")
        return candidate if candidate in self.states else None

    def sees(self, state: str, target: str) -> bool:
        return state in self.goals.get(target, ())


class NavigationEpisode:
    """One search: a scene, a target object, and a pose to start from.

    Rewards follow the original — `-0.01` a step and `+5` for calling `Done`
    while the target is visible. Calling `Done` when it is not ends the episode
    with no reward, so the agent has to commit rather than stall.
    """

    def __init__(self, scene: OfflineScene, target: str, state: str, max_steps: int = 30):
        self.scene = scene
        self.target = target
        self.state = state
        self.start_state = state
        self.max_steps = max_steps
        self.steps = 0
        self.done = False
        self.success = False

    @classmethod
    def sample(cls, scenes: Sequence[OfflineScene], rng: random.Random, max_steps: int = 30):
        """A random scene, one of its possible targets, and a random start pose."""
        while True:
            scene = rng.choice(scenes)
            targets = scene.targets
            if targets:
                break
        target = rng.choice(targets)
        # Starting where the target is already visible would teach nothing.
        states = [s for s in scene.states if not scene.sees(s, target)]
        return cls(scene, target, rng.choice(states or sorted(scene.states)), max_steps)

    def observation(self) -> np.ndarray:
        return self.scene.feature(self.state)

    def step(self, action_index: int) -> Tuple[float, bool, bool]:
        """Take an action. Returns `(reward, done, action_succeeded)`."""
        if self.done:
            raise RuntimeError("Episode is over; make a new one.")
        action = ACTIONS[action_index]
        self.steps += 1
        reward, succeeded = STEP_PENALTY, True

        if action == DONE:
            self.done = True
            self.success = self.scene.sees(self.state, self.target)
            if self.success:
                reward = GOAL_SUCCESS_REWARD
        else:
            nxt = self.scene.next_state(self.state, action)
            if nxt is None:
                succeeded = False  # walked into a wall; the pose does not change
            else:
                self.state = nxt

        if self.steps >= self.max_steps:
            self.done = True
        return reward, self.done, succeeded

    def optimal_steps(self) -> Optional[int]:
        """Shortest number of actions to a pose the target is visible from.

        Used for SPL, the standard measure that discounts a success by how far
        the agent wandered relative to the shortest route.
        """
        from collections import deque

        goals = self.scene.goals.get(self.target)
        if not goals:
            return None
        seen = {self.start_state}
        queue = deque([(self.start_state, 0)])
        while queue:
            state, distance = queue.popleft()
            if state in goals:
                return distance + 1  # plus the Done action itself
            for action in ACTIONS[:-1]:
                nxt = self.scene.next_state(state, action)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, distance + 1))
        return None


def load_scenes(
    root: str,
    rooms: Sequence[str] = tuple(ROOM_OBJECTS),
    split: str = "train",
    features_file: str = "resnet18_featuremap.hdf5",
) -> List[OfflineScene]:
    """Every scene of the given room types that is present under `root`."""
    scenes = []
    for name in scene_names(rooms, split):
        directory = os.path.join(root, name)
        if os.path.isdir(directory):
            scenes.append(OfflineScene(directory, features_file))
    if not scenes:
        raise FileNotFoundError(f"No scene folders found under {root}.")
    return scenes
