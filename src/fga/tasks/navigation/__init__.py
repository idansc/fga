"""Target-driven visual navigation.

An agent is told an object to find and attends its target embedding over the
spatial grid of its egocentric observation. The attended pair conditions a
recurrent policy emitting an action distribution and a value estimate — so unlike
the other use cases the output is a policy, trained by reinforcement learning
against episodes rather than by a supervised loss.

Episodes come from the SAVN offline dump of AI2-THOR: every reachable pose was
rendered once and its ResNet18 feature map cached, so training needs no simulator.
See [`environment`].

Ported from https://github.com/barmayo/spatial_attention.
"""

from .environment import (
    ACTIONS,
    GloveTargets,
    NavigationEpisode,
    OfflineScene,
    load_scenes,
)
from .modeling_navigation import (
    MODALITY_NAMES,
    NavigationConfig,
    NavigationOutput,
    NavigationPolicy,
)

__all__ = [
    "ACTIONS",
    "GloveTargets",
    "MODALITY_NAMES",
    "NavigationConfig",
    "NavigationEpisode",
    "NavigationOutput",
    "NavigationPolicy",
    "OfflineScene",
    "load_scenes",
]
