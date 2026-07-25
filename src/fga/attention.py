"""Factor-graph attention over an arbitrary list of utilities.

This is the reusable part of Factor Graph Attention (Schwartz et al., CVPR 2019).
It is deliberately independent of Visual Dialog: give it a list of utilities --
tensors of shape `(batch, num_entities, embed_dim)` -- and it returns one
attended vector per utility.

The attention logits for utility *i* are the sum of several *potentials*:

* **unary** -- how salient an entity is on its own,
* **self** -- how an entity relates to the other entities of the same utility,
* **pairwise** -- how an entity relates to the entities of every other utility,
* **prior** -- externally supplied bias (e.g. sentence-length cues).

The potentials are stacked along a channel axis and reduced by a learned,
bias-free `Conv1d(num_potentials -> 1)`, i.e. a learned weighting of factors,
then softmaxed.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Unary", "Pairwise", "Atten", "NaiveAttention", "Utility", "self_key", "pair_key"]


@dataclass(frozen=True)
class Utility:
    """A named modality handed to [`Atten`].

    This is the readable way to describe an attention graph. Compare

    ```python
    Atten(util_e=[512, 512, 128], sizes=[100, 21, 21],
          sharing_factor_weights={2: (9, [0, 1])})
    ```

    with

    ```python
    Atten.from_utilities([
        Utility("answer",  dim=512, size=100),
        Utility("question", dim=512, size=21),
        Utility("history", dim=128, size=21, repeats=9, connected_to=("answer", "question")),
    ])
    ```

    Args:
        name: identifier, used to declare connections and to label outputs.
        dim: embedding dimension of the utility's entities.
        size: number of entities. Required whenever pairwise factors are used,
            because they batch-norm over the flattened interaction grid.
        repeats: how many copies share one set of factor weights. `>1` means the
            input arrives as `(batch, repeats, entities, dim)` -- this is how the
            nine history rounds of Visual Dialog share parameters.
        connected_to: for a repeated utility, the names it interacts with. Only
            these pairs get factors, which is what keeps the history affordable.
            Two repeated utilities cannot be connected to each other.
    """

    name: str
    dim: int
    size: Optional[int] = None
    repeats: int = 1
    connected_to: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.repeats < 1:
            raise ValueError(f"Utility {self.name!r}: repeats must be >= 1, got {self.repeats}.")
        if self.repeats > 1 and not self.connected_to:
            raise ValueError(
                f"Utility {self.name!r} has repeats={self.repeats} but no connected_to. "
                "A shared utility must declare which utilities it interacts with."
            )
        if self.repeats == 1 and self.connected_to:
            raise ValueError(
                f"Utility {self.name!r} sets connected_to but repeats=1. Unshared utilities are "
                "connected to everything automatically."
            )


def utilities_to_legacy_args(
    utilities: Sequence[Utility],
) -> Tuple[List[int], List[Optional[int]], Dict[int, Tuple[int, List[int]]]]:
    """Translate [`Utility`] specs into the positional `(util_e, sizes, sharing)` form."""
    names = [u.name for u in utilities]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Utility names must be unique; repeated: {sorted(duplicates)}")

    index_of = {name: i for i, name in enumerate(names)}
    shared = {u.name for u in utilities if u.repeats > 1}

    sharing: Dict[int, Tuple[int, List[int]]] = {}
    for utility in utilities:
        if utility.repeats == 1:
            continue
        neighbours = []
        for neighbour in utility.connected_to:
            if neighbour not in index_of:
                raise ValueError(f"Utility {utility.name!r} is connected to unknown utility {neighbour!r}.")
            if neighbour in shared:
                raise ValueError(
                    f"Utility {utility.name!r} is connected to {neighbour!r}, but both share factor weights. "
                    "Connections between two shared utilities are not supported."
                )
            neighbours.append(index_of[neighbour])
        sharing[index_of[utility.name]] = (utility.repeats, neighbours)

    return [u.dim for u in utilities], [u.size for u in utilities], sharing


def self_key(idx: int) -> str:
    """ModuleDict key for a utility's self-interaction factor."""
    return f"self_{idx}"


def pair_key(idx1: int, idx2: int) -> str:
    """ModuleDict key for the factor between two distinct utilities."""
    return f"{idx1}_{idx2}"


class Unary(nn.Module):
    """Local potential: scores each entity of a utility on its own.

    Args:
        embed_size: embedding dimension of the utility.
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
    """Interaction potential between two utilities, or a utility and itself.

    Both utilities are projected to a common space, L2-normalized and multiplied
    to give a cosine-similarity grid `S`. When the spatial dimensions are known
    the grid is batch-normalized and marginalized by a learned `Conv1d`;
    otherwise it falls back to a plain mean over the opposite axis.

    Args:
        embed_x_size: embedding dimension of the first utility.
        x_spatial_dim: number of entities in the first utility, or `None` to use
            mean-marginalization instead of a learned one.
        embed_y_size: embedding dimension of the second utility. `None` for
            self-interaction.
        y_spatial_dim: number of entities in the second utility.
        self_interaction: this module scores a utility against itself, so only
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


class Atten(nn.Module):
    """Attention over a list of utilities, driven by factor-graph potentials.

    Prefer [`Atten.from_utilities`] with named [`Utility`] specs; the positional
    form below is what the paper used and is kept so that existing checkpoints
    and downstream forks keep working.

    Args:
        util_e: embedding dimension of each utility.
        sharing_factor_weights: maps a utility index to
            `(num_repeats, connected_utility_indices)`. Utilities listed here are
            expected with shape `(batch, num_repeats, entities, dim)` and reuse a
            single set of factor weights across the repeats -- this is how the
            history rounds of Visual Dialog share parameters. For efficiency,
            shared utilities are only connected to the utilities named in
            `connected_utility_indices`; connections *between* two shared
            utilities are not supported.
        use_prior: expect a `priors` argument in `forward`. Alias: `prior_flag`.
        sizes: number of entities per utility, needed for the batch-norm and the
            learned marginalization of the pairwise factors.
        size_force: adaptively average-pool utilities to `sizes` instead of
            requiring the inputs to already match them.
        use_pairwise: use interactions between distinct utilities. Alias: `pairwise_flag`.
        use_unary: use local information. Alias: `unary_flag`.
        use_self: use interactions among entities of the same utility. Alias: `self_flag`.
        unary_dropout: dropout inside the unary potential.
        legacy_unary_dropout: keep unary dropout active at evaluation time,
            reproducing the original release.
        utility_names: optional labels, set for you by [`Atten.from_utilities`].
    """

    def __init__(
        self,
        util_e: Sequence[int],
        sharing_factor_weights: Optional[Dict[int, Tuple[int, List[int]]]] = None,
        use_prior: bool = False,
        sizes: Optional[Sequence[Optional[int]]] = None,
        size_force: bool = False,
        use_pairwise: bool = True,
        use_unary: bool = True,
        use_self: bool = True,
        unary_dropout: float = 0.5,
        legacy_unary_dropout: bool = False,
        utility_names: Optional[Sequence[str]] = None,
        *,
        prior_flag: Optional[bool] = None,
        pairwise_flag: Optional[bool] = None,
        unary_flag: Optional[bool] = None,
        self_flag: Optional[bool] = None,
    ):
        super().__init__()
        # The `*_flag` spellings are the paper's; `use_*` matches the config and
        # the rest of the ecosystem. Both are accepted.
        use_prior = prior_flag if prior_flag is not None else use_prior
        use_pairwise = pairwise_flag if pairwise_flag is not None else use_pairwise
        use_unary = unary_flag if unary_flag is not None else use_unary
        use_self = self_flag if self_flag is not None else use_self

        self.util_e = list(util_e)
        self.n_utils = len(util_e)
        self.utility_names = (
            list(utility_names) if utility_names is not None else [str(i) for i in range(self.n_utils)]
        )
        if len(self.utility_names) != self.n_utils:
            raise ValueError(f"Got {len(self.utility_names)} utility_names for {self.n_utils} utilities.")

        self.spatial_pool = nn.ModuleDict()
        self.un_models = nn.ModuleList()

        self.use_prior = use_prior
        self.use_self = use_self
        self.use_pairwise = use_pairwise
        self.use_unary = use_unary
        self.size_force = size_force

        pairwise_flag = use_pairwise

        sizes = list(sizes) if sizes else [None for _ in util_e]
        self.sharing_factor_weights = dict(sharing_factor_weights or {})

        for idx, e_dim in enumerate(util_e):
            self.un_models.append(Unary(e_dim, unary_dropout, legacy_unary_dropout))
            if self.size_force:
                self.spatial_pool[str(idx)] = nn.AdaptiveAvgPool1d(sizes[idx])

        # Pairwise factors, keyed `self_i` and `i_j`. The original code keyed them
        # `"0"` and `"(0, 1)"`; parentheses and spaces are not valid attribute-path
        # segments, so `get_submodule` and `from_pretrained` choke on them.
        # `scripts/convert_legacy_checkpoint.py` translates the old names.
        self.pp_models = nn.ModuleDict()
        for (idx1, e_dim_1), (idx2, e_dim_2) in combinations_with_replacement(enumerate(util_e), 2):
            if self.use_self and idx1 == idx2:
                self.pp_models[self_key(idx1)] = Pairwise(e_dim_1, sizes[idx1], self_interaction=True)
            elif pairwise_flag:
                # Shared utilities are only wired to their declared neighbours.
                if idx1 in self.sharing_factor_weights and idx2 not in self.sharing_factor_weights[idx1][1]:
                    continue
                if idx2 in self.sharing_factor_weights and idx1 not in self.sharing_factor_weights[idx2][1]:
                    continue
                self.pp_models[pair_key(idx1, idx2)] = Pairwise(e_dim_1, sizes[idx1], e_dim_2, sizes[idx2])

        # One learned scalar mix per utility over however many potentials reach it.
        self.num_of_potentials: Dict[int, int] = {}
        self.default_num_of_potentials = sum([self.use_self, self.use_unary, self.use_prior])
        for idx in range(self.n_utils):
            self.num_of_potentials[idx] = self.default_num_of_potentials

        if pairwise_flag:
            for idx, (num_utils, connected_utils) in self.sharing_factor_weights.items():
                for c_u in connected_utils:
                    self.num_of_potentials[c_u] += num_utils
                    self.num_of_potentials[idx] += 1
            for k in self.num_of_potentials:
                if k not in self.sharing_factor_weights:
                    self.num_of_potentials[k] += (self.n_utils - 1) - len(self.sharing_factor_weights)

        self.reduce_potentials = nn.ModuleList(
            [nn.Conv1d(self.num_of_potentials[idx], 1, 1, bias=False) for idx in range(self.n_utils)]
        )

    @classmethod
    def from_utilities(cls, utilities: Sequence[Utility], **kwargs) -> "Atten":
        """Build from named [`Utility`] specs instead of parallel index-aligned lists.

        ```python
        attention = Atten.from_utilities(
            [
                Utility("answer", dim=512, size=100),
                Utility("question", dim=512, size=21),
                Utility("image", dim=2048, size=37),
                Utility("history", dim=128, size=21, repeats=9, connected_to=("answer", "question")),
            ],
            use_prior=True,
        )
        ```

        Any remaining keyword arguments go to [`Atten`].
        """
        util_e, sizes, sharing = utilities_to_legacy_args(utilities)
        return cls(
            util_e=util_e,
            sizes=sizes,
            sharing_factor_weights=sharing,
            utility_names=[u.name for u in utilities],
            **kwargs,
        )

    # The paper's spellings, kept so downstream forks that read these keep working.
    @property
    def prior_flag(self) -> bool:
        return self.use_prior

    @property
    def pairwise_flag(self) -> bool:
        return self.use_pairwise

    @property
    def unary_flag(self) -> bool:
        return self.use_unary

    @property
    def self_flag(self) -> bool:
        return self.use_self

    def describe(self) -> str:
        """A human-readable summary of the factor graph this module realizes."""
        lines = [f"Atten over {self.n_utils} utilities:"]
        for i, name in enumerate(self.utility_names):
            shared = self.sharing_factor_weights.get(i)
            detail = f"dim={self.util_e[i]}"
            if shared:
                neighbours = ", ".join(self.utility_names[j] for j in shared[1])
                detail += f", shared x{shared[0]} connected to [{neighbours}]"
            lines.append(f"  {i}. {name} ({detail}) <- {self.num_of_potentials[i]} potentials")
        return "\n".join(lines)

    def forward(
        self,
        utils: Sequence[torch.Tensor],
        priors: Optional[Sequence[Optional[torch.Tensor]]] = None,
        return_weights: bool = False,
    ):
        """Args:
            utils: one tensor per utility, `(batch, entities, dim)`, or
                `(batch, num_repeats, entities, dim)` for shared utilities.
            priors: one tensor per utility, `(batch, entities)`, or `None` entries
                for utilities without a prior. Required iff `prior_flag`.
            return_weights: also return the per-utility attention distributions,
                `(batch, entities)`, which is what you want for visualizations.

        Returns: attended representation per utility, `(batch, dim)`; a
            `(attention, weights)` tuple when `return_weights` is set.
        """
        assert self.n_utils == len(utils)
        assert (priors is None and not self.use_prior) or (
            priors is not None and self.use_prior and len(priors) == self.n_utils
        )
        # Copy so that callers keep ownership of their lists; `size_force` rewrites entries.
        utils = list(utils)
        priors = list(priors) if priors is not None else None

        b_size = utils[0].size(0)
        util_factors: Dict[int, List[torch.Tensor]] = {}
        attention: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []

        # Force a constant entity count; the pairwise batch-norm needs it.
        if self.size_force:
            for i, (num_utils, _) in self.sharing_factor_weights.items():
                if str(i) not in self.spatial_pool:
                    continue
                high_util = utils[i]
                high_util = high_util.view(num_utils * b_size, high_util.size(2), high_util.size(3))
                high_util = high_util.transpose(1, 2)
                utils[i] = self.spatial_pool[str(i)](high_util).transpose(1, 2)

            for i in range(self.n_utils):
                if i in self.sharing_factor_weights or str(i) not in self.spatial_pool:
                    continue
                utils[i] = self.spatial_pool[str(i)](utils[i].transpose(1, 2)).transpose(1, 2)
                if self.use_prior and priors[i] is not None:
                    priors[i] = self.spatial_pool[str(i)](priors[i].unsqueeze(1)).squeeze(1)

        # Shared-weight utilities (history rounds).
        for i, (num_utils, connected_list) in self.sharing_factor_weights.items():
            if self.use_unary:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.use_self:
                util_factors.setdefault(i, []).append(self.pp_models[self_key(i)](utils[i]))

            if self.use_pairwise:
                for j in connected_list:
                    other_util = utils[j]
                    expanded_util = (
                        other_util.unsqueeze(1)
                        .expand(b_size, num_utils, other_util.size(1), other_util.size(2))
                        .contiguous()
                        .view(b_size * num_utils, other_util.size(1), other_util.size(2))
                    )
                    if i < j:
                        factor_ij, factor_ji = self.pp_models[pair_key(i, j)](utils[i], expanded_util)
                    else:
                        factor_ji, factor_ij = self.pp_models[pair_key(j, i)](expanded_util, utils[i])
                    util_factors[i].append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji.view(b_size, num_utils, factor_ji.size(1)))

        # Local factors for the remaining utilities.
        for i in range(self.n_utils):
            if i in self.sharing_factor_weights:
                continue
            if self.use_unary:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.use_self:
                util_factors.setdefault(i, []).append(self.pp_models[self_key(i)](utils[i]))

        # Joint factors between the non-shared utilities.
        if self.use_pairwise:
            for i, j in combinations_with_replacement(range(self.n_utils), 2):
                if i in self.sharing_factor_weights or j in self.sharing_factor_weights or i == j:
                    continue
                factor_ij, factor_ji = self.pp_models[pair_key(i, j)](utils[i], utils[j])
                util_factors.setdefault(i, []).append(factor_ij)
                util_factors.setdefault(j, []).append(factor_ji)

        for i in range(self.n_utils):
            if self.use_prior:
                prior = priors[i] if priors[i] is not None else torch.zeros_like(util_factors[i][0])
                util_factors[i].append(prior)

            factors = torch.cat(
                [p if p.dim() == 3 else p.unsqueeze(1) for p in util_factors[i]],
                dim=1,
            )
            logits = self.reduce_potentials[i](factors).squeeze(1)
            weights = F.softmax(logits, dim=1)
            attention.append(torch.bmm(utils[i].transpose(1, 2), weights.unsqueeze(2)).squeeze(2))
            if return_weights:
                all_weights.append(weights)

        if return_weights:
            return attention, all_weights
        return attention


class NaiveAttention(nn.Module):
    """Ablation baseline: pools each utility with its prior, or uniformly."""

    def forward(self, utils, priors):
        atten = []
        spatial_atten = []
        for u, p in zip(utils, priors):
            if isinstance(u, tuple):
                u = u[1]
                num_elements = u.shape[0]
                if p is not None:
                    u = u.view(-1, u.shape[-2], u.shape[-1])
                    p = p.view(-1, p.shape[-2], p.shape[-1])
                    spatial_atten.append(
                        torch.bmm(p.transpose(1, 2), u).squeeze(2).view(num_elements, -1, u.shape[-2], u.shape[-1])
                    )
                else:
                    spatial_atten.append(u.mean(2))
                continue
            if p is not None:
                atten.append(torch.bmm(u.transpose(1, 2), p.unsqueeze(2)).squeeze(2))
            else:
                atten.append(u.mean(1))
        return atten, spatial_atten
