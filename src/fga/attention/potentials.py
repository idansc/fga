"""The potentials that make up a factor-graph attention.

Each potential scores the entities of a modality; [`FactorGraphAttention`] stacks
them and learns how much to weight each one.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Unary", "Pairwise", "self_key", "pair_key"]


def self_key(idx: int) -> str:
    """ModuleDict key for a modality's self-interaction factor."""
    return f"self_{idx}"


def pair_key(idx1: int, idx2: int) -> str:
    """ModuleDict key for the factor between two distinct modalities."""
    return f"{idx1}_{idx2}"


class Unary(nn.Module):
    """Local potential: scores each entity of a modality on its own.

    Args:
        embed_size: embedding dimension of the modality.
        dropout: dropout probability applied to the hidden activation.
        legacy_dropout: if `True`, keep dropping activations at evaluation time,
            reproducing the behaviour of the original release (which called
            `F.dropout` without forwarding `self.training`).
    """

    def __init__(self, embed_size: int, dropout: float = 0.5, legacy_dropout: bool = False):
        super().__init__()
        self.embed = nn.Conv1d(embed_size, embed_size, 1)
        self.feature_reduce = nn.Conv1d(embed_size, 1, 1)
        self.dropout = dropout
        self.legacy_dropout = legacy_dropout

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Args: X of shape `(batch, num_entities, embed_size)`.

        Returns: potentials of shape `(batch, num_entities)`.
        """
        X = X.transpose(1, 2)
        X_embed = self.embed(X)
        X_nl_embed = F.dropout(
            F.relu(X_embed),
            p=self.dropout,
            training=True if self.legacy_dropout else self.training,
        )
        X_poten = self.feature_reduce(X_nl_embed)
        return X_poten.squeeze(1)


class Pairwise(nn.Module):
    """Interaction potential between two modalities, or a modality and itself.

    Both modalities are projected to a common space, L2-normalized and multiplied
    to give a cosine-similarity grid `S`. When the spatial dimensions are known
    the grid is batch-normalized and marginalized by a learned `Conv1d`;
    otherwise it falls back to a plain mean over the opposite axis.

    Args:
        embed_x_size: embedding dimension of the first modality.
        x_spatial_dim: number of entities in the first modality, or `None` to use
            mean-marginalization instead of a learned one.
        embed_y_size: embedding dimension of the second modality. `None` for
            self-interaction.
        y_spatial_dim: number of entities in the second modality.
        self_interaction: this module scores a modality against itself, so only
            the `X` marginal is ever consumed. The original code still built and
            ran `margin_Y` here, discarding the result -- six wasted convolutions
            per forward pass and twelve parameters that never received a gradient.
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

        self.embed_X = nn.Conv1d(embed_x_size, self.embed_size, 1)
        self.embed_Y = nn.Conv1d(embed_y_size, self.embed_size, 1)
        if x_spatial_dim is not None:
            self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)
            self.margin_X = nn.Conv1d(self.y_spatial_dim, 1, 1)
            if not self_interaction:
                self.margin_Y = nn.Conv1d(self.x_spatial_dim, 1, 1)

    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None):
        """Args:
            X: `(batch, x_entities, embed_x_size)`.
            Y: `(batch, y_entities, embed_y_size)`, or `None` for self-interaction.

        Returns: `X_poten` alone when `Y is None`, else `(X_poten, Y_poten)`.
        """
        X_t = X.transpose(1, 2)
        Y_t = Y.transpose(1, 2) if Y is not None else X_t

        X_embed = self.embed_X(X_t)
        Y_embed = self.embed_Y(Y_t)

        X_norm = F.normalize(X_embed, dim=1)
        Y_norm = F.normalize(Y_embed, dim=1)

        S = X_norm.transpose(1, 2).bmm(Y_norm)
        if self.x_spatial_dim is not None:
            S = self.normalize_S(S.view(-1, self.x_spatial_dim * self.y_spatial_dim)).view(
                -1, self.x_spatial_dim, self.y_spatial_dim
            )
            X_poten = self.margin_X(S.transpose(1, 2)).transpose(1, 2).squeeze(2)
            if Y is None:
                return X_poten
            Y_poten = self.margin_Y(S).transpose(1, 2).squeeze(2)
        else:
            X_poten = S.mean(dim=2, keepdim=False)
            if Y is None:
                return X_poten
            Y_poten = S.mean(dim=1, keepdim=False)

        return X_poten, Y_poten
