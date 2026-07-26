"""Target-driven visual navigation, from https://github.com/barmayo/spatial_attention.

An agent is told an object to find and attends its target embedding over the
spatial grid of its egocentric observation. The attended pair conditions a
recurrent policy emitting an action distribution and a value estimate — so unlike
the other use cases the output is a policy, trained by reinforcement learning
against episodes rather than by a supervised loss.
"""

from .modeling_navigation import (
    MODALITY_NAMES,
    NavigationConfig,
    NavigationOutput,
    NavigationPolicy,
)

__all__ = ["MODALITY_NAMES", "NavigationConfig", "NavigationOutput", "NavigationPolicy"]
