"""Declaring the modalities an attention graph is built over.

The paper calls these *utilities*, so `Utility` remains an alias of
[`Modality`] and code written against the original API keeps working.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Modality", "Utility", "modalities_to_index_args", "utilities_to_index_args"]


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
        if self.repeats == 1 and self.connected_to:
            raise ValueError(
                f"Modality {self.name!r} sets connected_to but repeats=1. Unshared modalities are "
                "connected to everything automatically."
            )


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


#: The paper's term for the same concept; kept so existing code and the
#: downstream forks (simple-avsd, spatial_attention, VideoMatch) keep working.
Utility = Modality

#: Backwards-compatible alias.
utilities_to_index_args = modalities_to_index_args
