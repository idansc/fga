"""Factor-graph attention — the general, task-independent core.

This subpackage is deliberately free of any Visual Dialog code. It attends over
an arbitrary set of *modalities*, where a modality is any set of entities carrying
an embedding each: words in a sentence, regions in an image, frames in a video,
candidate answers, previous dialog rounds.

```python
from fga.attention import FactorGraphAttention, Modality

attention = FactorGraphAttention.from_modalities([
    Modality("text",  dim=512,  size=20),
    Modality("image", dim=2048, size=36),
])
pooled_text, pooled_image = attention([text_states, image_regions])
```

Modalities that repeat — the nine history rounds of Visual Dialog, the frames of a
video — share one set of factor weights via `repeats`, and declare which other
modalities they interact with via `connected_to`.

See `fga.tasks.visual_dialog` for a complete worked application.
"""

from .factor_graph import Atten, FactorGraphAttention, NaiveAttention
from .modality import Modality, Utility, modalities_to_index_args, utilities_to_index_args
from .potentials import Pairwise, Unary, pair_key, self_key

__all__ = [
    "Atten",
    "FactorGraphAttention",
    "NaiveAttention",
    "Modality",
    "Pairwise",
    "Unary",
    "Utility",
    "pair_key",
    "self_key",
    "modalities_to_index_args",
    "utilities_to_index_args",
]
