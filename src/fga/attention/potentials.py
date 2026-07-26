"""The potentials that make up a factor-graph attention.

Each potential scores the entities of a modality; [`FactorGraphAttention`] stacks
them and learns how much to weight each one.

Tensors are laid out `(batch, num_entities, channels)` throughout, so every
projection is a plain `nn.Linear` on the last axis. Earlier releases wrote these as
`Conv1d(kernel_size=1)` -- the same linear map in a `(batch, channels, length)`
layout, dispatched to the far slower convolution kernels. Checkpoints saved in
that shape convert once with `scripts/migrate_conv_checkpoint.py`.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Unary", "Pairwise", "Ternary", "self_key", "pair_key", "tri_key"]


class Unary(nn.Module):
    """Local potential: scores each entity of a modality on its own.

    Args:
        embed_size: embedding dimension of the modality.
        dropout: dropout probability applied to the hidden activation.

    Shape:
        - Input: `(batch, num_entities, embed_size)`
        - Output: `(batch, num_entities)`
    """

    def __init__(self, embed_size: int, dropout: float = 0.5):
        super().__init__()
        self.embed = nn.Linear(embed_size, embed_size)
        self.feature_reduce = nn.Linear(embed_size, 1)
        self.dropout = dropout

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        embedded = F.dropout(F.relu(self.embed(X)), p=self.dropout, training=self.training)
        return self.feature_reduce(embedded).squeeze(-1)


class Pairwise(nn.Module):
    """Interaction potential between two modalities, or a modality and itself.

    Both modalities are projected to a common space, L2-normalized and multiplied
    to give a cosine-similarity grid `S`. When the entity counts are known the grid
    is batch-normalized and marginalized by a learned linear map; otherwise it
    falls back to a plain mean over the opposite axis.

    Args:
        embed_x_size: embedding dimension of the first modality.
        x_spatial_dim: number of entities in the first modality, or `None` to use
            mean-marginalization instead of a learned one.
        embed_y_size: embedding dimension of the second modality. `None` for
            self-interaction.
        y_spatial_dim: number of entities in the second modality.
        self_interaction: this module scores a modality against itself, so only
            the `X` marginal is ever consumed. The original code still built and
            ran `margin_Y` here, discarding the result -- wasted work, and
            parameters that never received a gradient.

    Shape:
        - Input: `(batch, x_entities, embed_x_size)` and optionally
          `(batch, y_entities, embed_y_size)`
        - Output: `(batch, x_entities)`, or both marginals when `Y` is given.
    """

    def __init__(
        self,
        embed_x_size: int,
        x_spatial_dim: Optional[int] = None,
        embed_y_size: Optional[int] = None,
        y_spatial_dim: Optional[int] = None,
        self_interaction: bool = False,
    ):
        super().__init__()
        embed_y_size = embed_y_size if y_spatial_dim is not None else embed_x_size
        self.y_spatial_dim = y_spatial_dim if y_spatial_dim is not None else x_spatial_dim

        self.embed_size = max(embed_x_size, embed_y_size)
        self.x_spatial_dim = x_spatial_dim
        self.self_interaction = self_interaction

        self.embed_X = nn.Linear(embed_x_size, self.embed_size)
        self.embed_Y = nn.Linear(embed_y_size, self.embed_size)
        if x_spatial_dim is not None:
            self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)
            self.margin_X = nn.Linear(self.y_spatial_dim, 1)
            if not self_interaction:
                self.margin_Y = nn.Linear(self.x_spatial_dim, 1)

    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        """Returns `X_poten` alone when `Y is None`, else `(X_poten, Y_poten)`."""
        x = F.normalize(self.embed_X(X), dim=-1)
        y = F.normalize(self.embed_Y(Y if Y is not None else X), dim=-1)

        S = x @ y.transpose(1, 2)
        if self.x_spatial_dim is not None:
            S = self.normalize_S(S.reshape(-1, self.x_spatial_dim * self.y_spatial_dim)).view(
                -1, self.x_spatial_dim, self.y_spatial_dim
            )
            X_poten = self.margin_X(S).squeeze(-1)
            if Y is None:
                return X_poten
            Y_poten = self.margin_Y(S.transpose(1, 2)).squeeze(-1)
        else:
            X_poten = S.mean(dim=2)
            if Y is None:
                return X_poten
            Y_poten = S.mean(dim=1)

        return X_poten, Y_poten


def self_key(idx: int) -> str:
    """ModuleDict key for a modality's self-interaction factor."""
    return f"self_{idx}"


def pair_key(idx1: int, idx2: int) -> str:
    """ModuleDict key for the factor between two distinct modalities."""
    return f"{idx1}_{idx2}"


def tri_key(idx1: int, idx2: int, idx3: int) -> str:
    """ModuleDict key for the factor among three distinct modalities."""
    return f"tri_{idx1}_{idx2}_{idx3}"


class Ternary(nn.Module):
    """Three-way interaction potential, from High-Order Attention (NeurIPS 2017).

    Pairwise factors can only say "this region matches that word". A ternary
    factor scores triples directly -- "this region, this word *and* this candidate
    answer agree" -- which pairwise terms cannot express, since a triple can be
    jointly consistent while no two of its parts stand out alone.

    The three modalities are projected to a shared space, L2-normalized, and
    correlated into

        T[b, x, y, z] = sum_d  X[b, x, d] * Y[b, y, d] * Z[b, z, d]

    which is batch-normalized over the flattened grid and then marginalized down
    to one potential per modality: the potential for `X` collapses the `(y, z)`
    axes, and so on.

    This follows [`Pairwise`]'s conventions rather than the original Lua
    implementation's, so that a factor graph can mix the two: normalized
    embeddings and a batch-normalized grid instead of a learned elementwise
    scale, and no `tanh` on the output -- the potentials are combined by a learned
    reduction before the softmax, so squashing them here only discards range.

    Args:
        embed_size: shared projection dimension.
        x_size / y_size / z_size: entity counts, needed for the batch norm and the
            learned marginalizations.
        dim_x / dim_y / dim_z: input dimensions, when they differ from `embed_size`.

    Shape:
        - Input: `(batch, x_size, dim_x)`, `(batch, y_size, dim_y)`, `(batch, z_size, dim_z)`
        - Output: `(batch, x_size)`, `(batch, y_size)`, `(batch, z_size)`

    The interaction tensor holds `x_size * y_size * z_size` values per example, so
    it grows fast: the VQA configuration (196 regions, 15 words, 18 answers) is
    ~53k floats each, which is affordable, but a fourth modality would not be.
    """

    def __init__(
        self,
        embed_size: int,
        x_size: int,
        y_size: int,
        z_size: int,
        dim_x: Optional[int] = None,
        dim_y: Optional[int] = None,
        dim_z: Optional[int] = None,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.sizes = (x_size, y_size, z_size)

        self.embed_X = nn.Linear(dim_x or embed_size, embed_size)
        self.embed_Y = nn.Linear(dim_y or embed_size, embed_size)
        self.embed_Z = nn.Linear(dim_z or embed_size, embed_size)

        self.normalize_T = nn.BatchNorm1d(x_size * y_size * z_size)

        self.margin_X = nn.Linear(y_size * z_size, 1)
        self.margin_Y = nn.Linear(x_size * z_size, 1)
        self.margin_Z = nn.Linear(x_size * y_size, 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor):
        """Returns one potential per modality: `(X_poten, Y_poten, Z_poten)`."""
        x = F.normalize(self.embed_X(X), dim=-1)
        y = F.normalize(self.embed_Y(Y), dim=-1)
        z = F.normalize(self.embed_Z(Z), dim=-1)

        # One fused contraction rather than per-slice outer products.
        interaction = torch.einsum("bxd,byd,bzd->bxyz", x, y, z)

        batch = interaction.size(0)
        nx, ny, nz = self.sizes
        interaction = self.normalize_T(interaction.reshape(batch, nx * ny * nz)).view(batch, nx, ny, nz)

        return (
            self.margin_X(interaction.reshape(batch, nx, ny * nz)).squeeze(-1),
            self.margin_Y(interaction.permute(0, 2, 1, 3).reshape(batch, ny, nx * nz)).squeeze(-1),
            self.margin_Z(interaction.permute(0, 3, 1, 2).reshape(batch, nz, nx * ny)).squeeze(-1),
        )
