"""Factor Graph Attention -- a general multimodal attention module.

Reference implementation of "Factor Graph Attention" (Schwartz, Yu, Hazan and
Schwing, CVPR 2019), https://arxiv.org/abs/1904.05880, exposed through the
standard HuggingFace `transformers` interfaces.

```python
from fga import FGAConfig, FGAForVisualDialog

model = FGAForVisualDialog(FGAConfig())
model.save_pretrained("my-fga")
model = FGAForVisualDialog.from_pretrained("my-fga")
```
"""

from .attention import Atten, NaiveAttention, Pairwise, Unary, Utility
from .configuration_fga import FGAConfig
from .data import VisDialCollator, VisDialDataset, load_visdial_params, vocab_size_from_params
from .encoders import FGATextEncoder, LSTMTextEncoder, build_text_encoder, register_text_encoder
from .metrics import ndcg, scores_to_ranks, sparse_metrics
from .modeling_fga import (
    FGAForVisualDialog,
    FGAForVisualDialogOutput,
    FGAModel,
    FGAModelOutput,
)

__version__ = "1.0.0"

__all__ = [
    "Atten",
    "FGAConfig",
    "FGAForVisualDialog",
    "FGAForVisualDialogOutput",
    "FGAModel",
    "FGAModelOutput",
    "FGATextEncoder",
    "LSTMTextEncoder",
    "NaiveAttention",
    "Pairwise",
    "Unary",
    "Utility",
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
