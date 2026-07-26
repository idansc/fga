"""Declaring the modalities an attention graph is built over.

The paper calls these *utilities*, so `Utility` remains an alias of
[`Modality`] and code written against the original API keeps working.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Modality",
    "ModalityPlan",
    "Utility",
    "modalities_to_index_args",
    "plan_modalities",
    "utilities_to_index_args",
]


@dataclass(frozen=True)
class Modality:
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
            Two repeated modalities cannot be connected to each other.
    """

    name: str
    dim: int
    size: Optional[int] = None
    repeats: int = 1
    connected_to: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.repeats < 1:
            raise ValueError(f"Modality {self.name!r}: repeats must be >= 1, got {self.repeats}.")
        if self.repeats > 1 and not self.connected_to:
            raise ValueError(
                f"Modality {self.name!r} has repeats={self.repeats} but no connected_to. "
                "A shared modality must declare which utilities it interacts with."
            )
        # `connected_to` on an untied, unrepeated modality is allowed and simply
        # redundant -- such a modality already interacts with everything. It only
        # constrains anything for a modality that shares weights, whether via
        # `repeats` or via a `tied_weights` group.


def modalities_to_index_args(
    modalities: Sequence[Modality],
) -> Tuple[List[int], List[Optional[int]], Dict[int, Tuple[int, List[int]]]]:
    """Translate [`Modality`] specs into the positional `(util_e, sizes, sharing)` form."""
    names = [u.name for u in modalities]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Modality names must be unique; repeated: {sorted(duplicates)}")

    index_of = {name: i for i, name in enumerate(names)}
    shared = {u.name for u in modalities if u.repeats > 1}

    sharing: Dict[int, Tuple[int, List[int]]] = {}
    for modality in modalities:
        if modality.repeats == 1:
            continue
        neighbours = []
        for neighbour in modality.connected_to:
            if neighbour not in index_of:
                raise ValueError(f"Modality {modality.name!r} is connected to unknown modality {neighbour!r}.")
            if neighbour in shared:
                raise ValueError(
                    f"Modality {modality.name!r} is connected to {neighbour!r}, but both share factor weights. "
                    "Connections between two shared utilities are not supported."
                )
            neighbours.append(index_of[neighbour])
        sharing[index_of[modality.name]] = (modality.repeats, neighbours)

    return [u.dim for u in modalities], [u.size for u in modalities], sharing


@dataclass(frozen=True)
class ModalityPlan:
    """How a user-facing list of modalities maps onto the shared-weight machinery.

    Modalities that share factor weights are one *group*. Internally a group is a
    single entry carrying `len(group)` repeats, because the weights are shared;
    externally it stays N separate modalities the caller passes and receives
    individually. This object is the translation between the two.

    Attributes:
        embed_dims / num_entities / sharing_factor_weights: the positional form
            [`FactorGraphAttention`] is built from.
        groups: for each internal entry, the user-facing modality indices it
            covers -- a single index for an untied modality, N for a tied group.
        names: user-facing modality names, in the order they are passed.
    """

    embed_dims: Tuple[int, ...]
    num_entities: Tuple[Optional[int], ...]
    sharing_factor_weights: Dict[int, Tuple[int, List[int]]]
    groups: Tuple[Tuple[int, ...], ...]
    names: Tuple[str, ...]

    @property
    def is_trivial(self) -> bool:
        """True when every group holds exactly one modality, so no regrouping is needed."""
        return all(len(group) == 1 for group in self.groups)


def plan_modalities(
    modalities: Sequence[Modality],
    tied_weights: Optional[Sequence[Sequence[str]]] = None,
) -> ModalityPlan:
    """Work out the internal layout for a list of modalities and their tied groups.

    Args:
        modalities: the modalities, in the order they will be passed to `forward`.
        tied_weights: groups of modality names that share one set of factor
            weights, e.g. `[("history_1", ..., "history_9")]`. Members must agree
            on dimension, entity count and connections, since they are literally
            the same weights.

    A group is placed at the position of its first member, so the caller's
    ordering is preserved.
    """
    names = [m.name for m in modalities]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Modality names must be unique; repeated: {sorted(duplicates)}")

    by_name = {m.name: m for m in modalities}
    index_of = {name: i for i, name in enumerate(names)}

    groups: List[List[int]] = []
    grouped: Dict[str, int] = {}
    for group in tied_weights or []:
        members = list(group)
        if len(members) < 2:
            raise ValueError(f"A tied_weights group needs at least two modalities; got {members}.")
        for name in members:
            if name not in by_name:
                raise ValueError(f"tied_weights names unknown modality {name!r}.")
            if name in grouped:
                raise ValueError(f"Modality {name!r} appears in more than one tied_weights group.")

        first = by_name[members[0]]
        for name in members[1:]:
            other = by_name[name]
            if (other.dim, other.size) != (first.dim, first.size):
                raise ValueError(
                    f"Tied modalities must match in shape: {name!r} is "
                    f"(dim={other.dim}, size={other.size}) but {first.name!r} is "
                    f"(dim={first.dim}, size={first.size}). They share the same weights."
                )
            if tuple(other.connected_to or ()) != tuple(first.connected_to or ()):
                raise ValueError(
                    f"Tied modalities must share connections: {name!r} connects to "
                    f"{other.connected_to} but {first.name!r} connects to {first.connected_to}."
                )
        group_id = len(groups)
        for name in members:
            grouped[name] = group_id
        groups.append([index_of[name] for name in members])

    # Lay out the internal entries, keeping the caller's order.
    entries: List[List[int]] = []
    seen_groups = set()
    for i, modality in enumerate(modalities):
        group_id = grouped.get(modality.name)
        if group_id is None:
            entries.append([i])
        elif group_id not in seen_groups:
            seen_groups.add(group_id)
            entries.append(groups[group_id])

    entry_of_modality = {i: e for e, members in enumerate(entries) for i in members}
    tied_names = set(grouped)

    sharing: Dict[int, Tuple[int, List[int]]] = {}
    for entry, members in enumerate(entries):
        representative = modalities[members[0]]
        repeats = len(members) if len(members) > 1 else representative.repeats
        if repeats == 1:
            continue
        neighbours = []
        for neighbour in representative.connected_to or ():
            if neighbour not in index_of:
                raise ValueError(f"Modality {representative.name!r} is connected to unknown modality {neighbour!r}.")
            if neighbour in tied_names or by_name[neighbour].repeats > 1:
                raise ValueError(
                    f"Modality {representative.name!r} is connected to {neighbour!r}, but both share "
                    "factor weights. Connections between two shared modalities are not supported."
                )
            neighbours.append(entry_of_modality[index_of[neighbour]])
        sharing[entry] = (repeats, neighbours)

    return ModalityPlan(
        embed_dims=tuple(modalities[m[0]].dim for m in entries),
        num_entities=tuple(modalities[m[0]].size for m in entries),
        sharing_factor_weights=sharing,
        groups=tuple(tuple(m) for m in entries),
        names=tuple(names),
    )


#: The paper's term for the same concept; kept so existing code and the
#: downstream forks (simple-avsd, spatial_attention, VideoMatch) keep working.
Utility = Modality

#: Backwards-compatible alias.
utilities_to_index_args = modalities_to_index_args
