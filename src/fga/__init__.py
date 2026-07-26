"""Factor Graph Attention — a general multimodal attention layer.

Reference implementation of "Factor Graph Attention" (Schwartz, Yu, Hazan and
Schwing, CVPR 2019), https://arxiv.org/abs/1904.05880.

The package is in two halves. [`fga.attention`] is the general layer: it attends
over any set of *utilities* — words, image regions, video frames, candidate
answers — and carries no assumptions about a task. It is used like any other
`torch.nn` module:

```python
from fga import FactorGraphAttention

attention = FactorGraphAttention(embed_dims=[512, 2048], num_entities=[20, 36])
pooled_text, pooled_image = attention(text, image)
```

[`fga.tasks.visual_dialog`] is the worked application the paper reports, exposed
through the standard `transformers` interfaces:

```python
from fga import FGAForVisualDialog

model = FGAForVisualDialog.from_pretrained("Idan/fga")
```
"""

from .attention import (
    Atten,
    FactorGraphAttention,
    Modality,
    NaiveAttention,
    Pairwise,
    Unary,
    Utility,
)
from .tasks.visual_dialog import (
    FGAConfig,
    FGAForVisualDialog,
    FGAForVisualDialogOutput,
    FGAModel,
    FGAModelOutput,
    FGATextEncoder,
    LSTMTextEncoder,
    VisDialCollator,
    VisDialDataset,
    build_text_encoder,
    load_visdial_params,
    ndcg,
    register_text_encoder,
    scores_to_ranks,
    sparse_metrics,
    vocab_size_from_params,
)

__version__ = "1.1.0"

__all__ = [
    # the general layer
    "Atten",
    "FactorGraphAttention",
    "NaiveAttention",
    "Modality",
    "Pairwise",
    "Unary",
    "Utility",
    # the visual dialog application
    "FGAConfig",
    "FGAForVisualDialog",
    "FGAForVisualDialogOutput",
    "FGAModel",
    "FGAModelOutput",
    "FGATextEncoder",
    "LSTMTextEncoder",
    "VisDialCollator",
    "VisDialDataset",
    "build_text_encoder",
    "load_visdial_params",
    "ndcg",
    "register_text_encoder",
    "scores_to_ranks",
    "sparse_metrics",
    "vocab_size_from_params",
    "__version__",
]


def _register_for_auto_classes() -> None:
    """Make `AutoConfig`/`AutoModel` resolve `model_type="fga"` locally.

    Lets `AutoModel.from_pretrained(...)` work for anyone who has imported `fga`,
    without the model needing to live in the `transformers` codebase.
    """
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:  # pragma: no cover - transformers is a hard dependency
        return

    try:
        AutoConfig.register(FGAConfig.model_type, FGAConfig)
        AutoModel.register(FGAConfig, FGAModel)
    except ValueError:
        # Already registered, e.g. when the module is reloaded.
        pass


_register_for_auto_classes()
