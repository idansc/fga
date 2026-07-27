"""Layers the VQA models share.

----

`GatedTanh` follows "Tips and Tricks for Visual Question Answering: Learnings
from the 2017 Challenge" (Teney, Anderson, He and van den Hengel, CVPR 2018).
"""

import torch
import torch.nn as nn

__all__ = ["GatedTanh"]


class GatedTanh(nn.Module):
    r"""A `tanh` transform with a learned multiplicative gate.

    .. math::
        y = \tanh(Wx + b) \odot \sigma(W'x + b')

    Two projections instead of one: the first proposes a value, the second
    decides, per dimension, how much of it to let through. That lets a unit stay
    silent for inputs it has nothing to say about rather than being forced to a
    saturating value, which is the failure mode of a plain `tanh` on features
    whose scale it did not expect. The 2017 challenge entry found this the most
    useful of its non-linearities.

    Args:
        in_features: input dimension.
        out_features: output dimension.

    Shape:
        - Input: `(*, in_features)`
        - Output: `(*, out_features)`
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.value = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.value(x)) * torch.sigmoid(self.gate(x))

    def extra_repr(self) -> str:
        return f"in_features={self.value.in_features}, out_features={self.value.out_features}"
