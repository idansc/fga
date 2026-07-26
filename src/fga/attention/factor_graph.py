"""Factor-graph attention over an arbitrary set of modalities.

This is the general, task-independent core of Factor Graph Attention
(Schwartz et al., CVPR 2019). It knows nothing about Visual Dialog: give it a
list of modalities -- tensors of shape `(batch, num_entities, embed_dim)` -- and
it returns one attended vector per modality.

The attention logits for modality *i* are the sum of several *potentials*:

* **unary** -- how salient an entity is on its own,
* **self** -- how an entity relates to the other entities of the same modality,
* **pairwise** -- how an entity relates to the entities of every other modality,
* **prior** -- externally supplied bias (e.g. sentence-length cues).

The potentials are stacked along a channel axis and reduced by a learned,
bias-free `Conv1d(num_potentials -> 1)`, i.e. a learned weighting of factors,
then softmaxed.
"""

from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modality import Modality, ModalityPlan, plan_modalities
from .potentials import Pairwise, Unary, conv1x1, pair_key, self_key

__all__ = ["FactorGraphAttention", "Atten", "NaiveAttention"]


class FactorGraphAttention(nn.Module):
    r"""Multimodal attention over a set of modalities, as a standard `nn` layer.

    A *modality* is any set of entities carrying an embedding each: words in a
    sentence, regions in an image, frames in a video, candidate answers. The
    layer attends over all of them jointly and returns one pooled vector per
    modality. (The paper calls these *utilities*.)

    Used like any other module:

    ```python
    import torch
    from fga.attention import FactorGraphAttention

    attention = FactorGraphAttention(embed_dims=[512, 2048], num_entities=[20, 36])
    text = torch.randn(8, 20, 512)
    image = torch.randn(8, 36, 2048)
    pooled_text, pooled_image = attention(text, image)   # (8, 512), (8, 2048)
    ```

    Shape:
        - Input: one tensor per modality, `(batch, num_entities_i, embed_dim_i)`;
          a repeated modality takes `(batch * repeats, num_entities_i, embed_dim_i)`.
        - Output: a list of `(batch, embed_dim_i)`.

    Args:
        embed_dims: embedding dimension of each modality. Alias: `util_e`.
        num_entities: number of entities per modality. Needed whenever pairwise
            factors are used, since they normalize over the flattened
            interaction grid. Alias: `sizes`.
        sharing_factor_weights: maps a modality index to
            `(num_repeats, connected_modality_indices)`. Modalities listed here are
            expected with shape `(batch, num_repeats, entities, dim)` and reuse a
            single set of factor weights across the repeats -- this is how the
            history rounds of Visual Dialog share parameters. For efficiency,
            shared modalities are only connected to the modalities named in
            `connected_modality_indices`; connections *between* two shared
            modalities are not supported. Prefer declaring this by name with
            [`FactorGraphAttention.from_modalities`].
        use_prior: expect a `priors` argument in `forward`. Alias: `prior_flag`.
        size_force: adaptively average-pool modalities to `num_entities` instead
            of requiring the inputs to already match them.
        use_pairwise: use interactions between distinct modalities. Alias: `pairwise_flag`.
        use_unary: use local information. Alias: `unary_flag`.
        use_self: use interactions among entities of the same modality. Alias: `self_flag`.
        unary_dropout: dropout inside the unary potential.
        legacy_unary_dropout: keep unary dropout active at evaluation time,
            reproducing the original release.
        modality_names: optional labels, set for you by [`FactorGraphAttention.from_modalities`].
    """

    def __init__(
        self,
        embed_dims: Optional[Sequence[int]] = None,
        num_entities: Optional[Sequence[Optional[int]]] = None,
        sharing_factor_weights: Optional[Dict[int, Tuple[int, List[int]]]] = None,
        use_prior: bool = False,
        size_force: bool = False,
        use_pairwise: bool = True,
        use_unary: bool = True,
        use_self: bool = True,
        unary_dropout: float = 0.5,
        legacy_unary_dropout: bool = False,
        modality_names: Optional[Sequence[str]] = None,
        *,
        util_e: Optional[Sequence[int]] = None,
        sizes: Optional[Sequence[Optional[int]]] = None,
        prior_flag: Optional[bool] = None,
        pairwise_flag: Optional[bool] = None,
        unary_flag: Optional[bool] = None,
        self_flag: Optional[bool] = None,
        high_order_utils: Optional[Sequence[Tuple[int, int, Sequence[int]]]] = None,
        size_flag: Optional[bool] = None,
        utility_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        # `util_e` / `sizes` are the paper's names, used by the published
        # checkpoints and the downstream forks; `embed_dims` / `num_entities`
        # say the same thing in the vocabulary of `torch.nn`.
        modality_names = utility_names if utility_names is not None else modality_names
        embed_dims = util_e if util_e is not None else embed_dims
        sizes = sizes if sizes is not None else num_entities
        if embed_dims is None:
            raise TypeError("FactorGraphAttention requires embed_dims (one embedding size per modality).")
        util_e = embed_dims

        # The AVSD fork (idansc/simple-avsd) calls the shared-weight modalities
        # `high_order_utils` and passes a list of `(index, repeats, connected)`
        # triples rather than a dict, and gates the entity counts behind
        # `size_flag`. Accepting both makes this package a drop-in for that code.
        if high_order_utils:
            if sharing_factor_weights:
                raise ValueError("Pass either high_order_utils or sharing_factor_weights, not both.")
            sharing_factor_weights = {int(i): (int(n), list(connected)) for i, n, connected in high_order_utils}
        if size_flag is False:
            sizes = None
        # The `*_flag` spellings are the paper's; `use_*` matches the config and
        # the rest of the ecosystem. Both are accepted.
        use_prior = prior_flag if prior_flag is not None else use_prior
        use_pairwise = pairwise_flag if pairwise_flag is not None else use_pairwise
        use_unary = unary_flag if unary_flag is not None else use_unary
        use_self = self_flag if self_flag is not None else use_self

        self.util_e = list(util_e)
        self.n_modalities = len(util_e)
        self.modality_names = (
            list(modality_names) if modality_names is not None else [str(i) for i in range(self.n_modalities)]
        )
        if len(self.modality_names) != self.n_modalities:
            raise ValueError(f"Got {len(self.modality_names)} modality_names for {self.n_modalities} modalities.")

        # Set by `from_modalities` when modalities are grouped into shared-weight
        # groups; `None` means the caller passes one tensor per internal entry.
        self._plan: Optional[ModalityPlan] = None

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
        for idx in range(self.n_modalities):
            self.num_of_potentials[idx] = self.default_num_of_potentials

        if pairwise_flag:
            for idx, (num_utils, connected_utils) in self.sharing_factor_weights.items():
                for c_u in connected_utils:
                    self.num_of_potentials[c_u] += num_utils
                    self.num_of_potentials[idx] += 1
            for k in self.num_of_potentials:
                if k not in self.sharing_factor_weights:
                    self.num_of_potentials[k] += (self.n_modalities - 1) - len(self.sharing_factor_weights)

        self.reduce_potentials = nn.ModuleList(
            [nn.Conv1d(self.num_of_potentials[idx], 1, 1, bias=False) for idx in range(self.n_modalities)]
        )

    @classmethod
    def from_modalities(
        cls,
        modalities: Sequence[Modality],
        share_weights: Optional[Sequence[Sequence[str]]] = None,
        **kwargs,
    ) -> "FactorGraphAttention":
        """Build from named [`Modality`] specs instead of parallel index-aligned lists.

        ```python
        attention = FactorGraphAttention.from_modalities(
            [
                Modality("answer", dim=512, size=100),
                Modality("question", dim=512, size=21),
                Modality("image", dim=2048, size=37),
                Modality("history", dim=128, size=21, repeats=9, connected_to=("answer", "question")),
            ],
            use_prior=True,
        )
        ```

        Any remaining keyword arguments go to [`FactorGraphAttention`].
        """
        plan = plan_modalities(modalities, share_weights)
        module = cls(
            embed_dims=list(plan.embed_dims),
            num_entities=list(plan.num_entities),
            sharing_factor_weights=plan.sharing_factor_weights,
            modality_names=[modalities[group[0]].name for group in plan.groups],
            **kwargs,
        )
        module._plan = plan
        return module

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

    def extra_repr(self) -> str:
        """Summarize the graph in `print(model)`, as `nn` layers do."""
        parts = [
            "modalities=[" + ", ".join(f"{n}:{d}" for n, d in zip(self.modality_names, self.util_e)) + "]",
            "factors="
            + "+".join(
                name
                for name, on in (
                    ("unary", self.use_unary),
                    ("self", self.use_self),
                    ("pairwise", self.use_pairwise),
                    ("prior", self.use_prior),
                )
                if on
            ),
        ]
        if self.sharing_factor_weights:
            shared = ", ".join(
                f"{self.modality_names[i]}x{n}" for i, (n, _) in sorted(self.sharing_factor_weights.items())
            )
            parts.append(f"shared=[{shared}]")
        return ", ".join(parts)

    def describe(self) -> str:
        """A human-readable summary of the factor graph this module realizes."""
        lines = [f"{type(self).__name__} over {self.n_modalities} modalities:"]
        for i, name in enumerate(self.modality_names):
            shared = self.sharing_factor_weights.get(i)
            detail = f"dim={self.util_e[i]}"
            if shared:
                neighbours = ", ".join(self.modality_names[j] for j in shared[1])
                detail += f", shared x{shared[0]} connected to [{neighbours}]"
            lines.append(f"  {i}. {name} ({detail}) <- {self.num_of_potentials[i]} potentials")
        return "\n".join(lines)

    def forward(
        self,
        *modalities,
        priors: Optional[Sequence[Optional[torch.Tensor]]] = None,
        return_weights: bool = False,
    ):
        """Attend over the modalities.

        Modalities may be passed either positionally or as a single sequence, so
        both of these work:

        ```python
        pooled_text, pooled_image = attention(text, image)
        pooled = attention([text, image])
        ```

        Args:
            modalities: one tensor per modality, `(batch, entities, dim)`, or
                `(batch * repeats, entities, dim)` for a shared modality.
            priors: one tensor per modality, `(batch, entities)`, or `None` entries
                for modalities without a prior. Required iff `use_prior`.
            return_weights: also return the per-modality attention distributions,
                `(batch, entities)`, which is what you want for visualizations.

        Returns: attended representation per modality, `(batch, dim)`; a
            `(attention, weights)` tuple when `return_weights` is set.
        """
        # A single list/tuple argument is the sequence form.
        if len(modalities) == 1 and isinstance(modalities[0], (list, tuple)):
            modalities = modalities[0]

        plan = self._plan
        if plan is not None and not plan.is_trivial and len(modalities) == len(plan.names):
            # The caller passes one tensor per *modality*; modalities sharing weights use a
            # single set of weights, so their tensors are merged into one entry.
            # The shared factors index rows as batch-major, repeat-minor -- the same
            # order `expand(batch, repeats, ...).view(batch * repeats, ...)` produces --
            # so members must be stacked along a new axis 1, not concatenated.
            merged = []
            for group in plan.groups:
                if len(group) == 1:
                    merged.append(modalities[group[0]])
                else:
                    stacked = torch.stack([modalities[i] for i in group], dim=1)
                    merged.append(stacked.reshape(-1, *stacked.shape[2:]))
            if priors is not None:
                merged_priors = []
                for group in plan.groups:
                    if len(group) == 1:
                        merged_priors.append(priors[group[0]])
                    elif priors[group[0]] is None:
                        merged_priors.append(None)
                    else:
                        stacked = torch.stack([priors[i] for i in group], dim=1)
                        merged_priors.append(stacked.reshape(-1, *stacked.shape[2:]))
                priors = merged_priors
            modalities = merged

            result = self._attend(modalities, priors, return_weights)
            attention, weights = result if return_weights else (result, None)

            # Split the shared entries back out, in the caller's modality order.
            per_modality: List[Optional[torch.Tensor]] = [None] * len(plan.names)
            per_modality_weights: List[Optional[torch.Tensor]] = [None] * len(plan.names)
            for entry, group in enumerate(plan.groups):
                if len(group) == 1:
                    per_modality[group[0]] = attention[entry]
                    if weights is not None:
                        per_modality_weights[group[0]] = weights[entry]
                else:
                    pooled = attention[entry].view(-1, len(group), attention[entry].size(-1))
                    for position, index in enumerate(group):
                        per_modality[index] = pooled[:, position]
                    if weights is not None:
                        w = weights[entry].view(-1, len(group), weights[entry].size(-1))
                        for position, index in enumerate(group):
                            per_modality_weights[index] = w[:, position]

            if return_weights:
                return per_modality, per_modality_weights
            return per_modality

        return self._attend(modalities, priors, return_weights)

    def _attend(self, modalities, priors, return_weights: bool):
        """Attend over one tensor per internal entry, shared groups already merged."""
        if self.n_modalities != len(modalities):
            raise ValueError(
                f"{type(self).__name__} was built for {self.n_modalities} utilities "
                f"({', '.join(self.modality_names)}) but got {len(modalities)}."
            )
        assert (priors is None and not self.use_prior) or (
            priors is not None and self.use_prior and len(priors) == self.n_modalities
        )
        # Copy so that callers keep ownership of their lists; `size_force` rewrites entries.
        modalities = list(modalities)
        priors = list(priors) if priors is not None else None

        b_size = modalities[0].size(0)
        util_factors: Dict[int, List[torch.Tensor]] = {}
        attention: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []

        # Force a constant entity count; the pairwise batch-norm needs it.
        if self.size_force:
            for i, (num_utils, _) in self.sharing_factor_weights.items():
                if str(i) not in self.spatial_pool:
                    continue
                high_util = modalities[i]
                high_util = high_util.view(num_utils * b_size, high_util.size(2), high_util.size(3))
                high_util = high_util.transpose(1, 2)
                modalities[i] = self.spatial_pool[str(i)](high_util).transpose(1, 2)

            for i in range(self.n_modalities):
                if i in self.sharing_factor_weights or str(i) not in self.spatial_pool:
                    continue
                modalities[i] = self.spatial_pool[str(i)](modalities[i].transpose(1, 2)).transpose(1, 2)
                if self.use_prior and priors[i] is not None:
                    priors[i] = self.spatial_pool[str(i)](priors[i].unsqueeze(1)).squeeze(1)

        # Shared-weight utilities (history rounds).
        for i, (num_utils, connected_list) in self.sharing_factor_weights.items():
            if self.use_unary:
                util_factors.setdefault(i, []).append(self.un_models[i](modalities[i]))
            if self.use_self:
                util_factors.setdefault(i, []).append(self.pp_models[self_key(i)](modalities[i]))

            if self.use_pairwise:
                for j in connected_list:
                    other_util = modalities[j]
                    expanded_util = (
                        other_util.unsqueeze(1)
                        .expand(b_size, num_utils, other_util.size(1), other_util.size(2))
                        .contiguous()
                        .view(b_size * num_utils, other_util.size(1), other_util.size(2))
                    )
                    if i < j:
                        factor_ij, factor_ji = self.pp_models[pair_key(i, j)](modalities[i], expanded_util)
                    else:
                        factor_ji, factor_ij = self.pp_models[pair_key(j, i)](expanded_util, modalities[i])
                    util_factors[i].append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji.view(b_size, num_utils, factor_ji.size(1)))

        # Local factors for the remaining utilities.
        for i in range(self.n_modalities):
            if i in self.sharing_factor_weights:
                continue
            if self.use_unary:
                util_factors.setdefault(i, []).append(self.un_models[i](modalities[i]))
            if self.use_self:
                util_factors.setdefault(i, []).append(self.pp_models[self_key(i)](modalities[i]))

        # Joint factors between the non-shared utilities.
        if self.use_pairwise:
            for i, j in combinations_with_replacement(range(self.n_modalities), 2):
                if i in self.sharing_factor_weights or j in self.sharing_factor_weights or i == j:
                    continue
                factor_ij, factor_ji = self.pp_models[pair_key(i, j)](modalities[i], modalities[j])
                util_factors.setdefault(i, []).append(factor_ij)
                util_factors.setdefault(j, []).append(factor_ji)

        for i in range(self.n_modalities):
            if self.use_prior:
                prior = priors[i] if priors[i] is not None else torch.zeros_like(util_factors[i][0])
                util_factors[i].append(prior)

            factors = torch.cat(
                [p if p.dim() == 3 else p.unsqueeze(1) for p in util_factors[i]],
                dim=1,
            )
            logits = conv1x1(factors, self.reduce_potentials[i]).squeeze(1)
            weights = F.softmax(logits, dim=1)
            attention.append(torch.bmm(modalities[i].transpose(1, 2), weights.unsqueeze(2)).squeeze(2))
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


#: The name used in the paper and in every downstream fork.
Atten = FactorGraphAttention

# The paper's vocabulary, kept as aliases.
FactorGraphAttention.from_utilities = FactorGraphAttention.from_modalities
FactorGraphAttention.utility_names = property(lambda self: self.modality_names)
FactorGraphAttention.n_utils = property(lambda self: self.n_modalities)
