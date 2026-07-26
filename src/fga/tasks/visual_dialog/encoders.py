"""Text encoders that turn token ids into per-token utility embeddings.

FGA is agnostic to how text becomes a sequence of vectors: the attention module
only needs `(batch, num_tokens, dim)` tensors. The paper used one LSTM per field
over a vocabulary built from the VisDial h5 dump, which is what
[`LSTMTextEncoder`] implements and what the released checkpoints contain.

To plug in a different encoder -- a pretrained transformer, say -- subclass
[`FGATextEncoder`], register it, and point `config.text_encoder_type` at it:

```python
from fga.encoders import FGATextEncoder, register_text_encoder

@register_text_encoder("my_encoder")
class MyEncoder(FGATextEncoder):
    def __init__(self, config): ...
    def forward(self, input_ids, field): ...
```

Note that a transformer encoder needs raw text, so it also needs a data pipeline
that tokenizes the VisDial JSON rather than reading the pre-indexed h5 file.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type

import torch
import torch.nn as nn

__all__ = [
    "FIELDS",
    "FGATextEncoder",
    "LSTMTextEncoder",
    "register_text_encoder",
    "build_text_encoder",
]

#: The text fields FGA encodes, in no particular order.
FIELDS = ("question", "answer", "caption", "history_question", "history_answer")

TEXT_ENCODER_REGISTRY: Dict[str, Type["FGATextEncoder"]] = {}


def register_text_encoder(name: str) -> Callable[[Type["FGATextEncoder"]], Type["FGATextEncoder"]]:
    """Class decorator registering an encoder under `config.text_encoder_type`."""

    def decorator(cls: Type["FGATextEncoder"]) -> Type["FGATextEncoder"]:
        if name in TEXT_ENCODER_REGISTRY:
            raise ValueError(f"Text encoder '{name}' is already registered.")
        TEXT_ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def build_text_encoder(config) -> "FGATextEncoder":
    """Instantiate the encoder named by `config.text_encoder_type`."""
    try:
        cls = TEXT_ENCODER_REGISTRY[config.text_encoder_type]
    except KeyError:
        raise ValueError(
            f"Unknown text_encoder_type '{config.text_encoder_type}'. "
            f"Registered encoders: {sorted(TEXT_ENCODER_REGISTRY)}."
        ) from None
    return cls(config)


class FGATextEncoder(nn.Module, ABC):
    """Maps token ids of a given field to a sequence of contextual vectors."""

    @property
    @abstractmethod
    def output_dims(self) -> Dict[str, int]:
        """Output dimension per field name."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, field: str) -> torch.Tensor:
        """Args:
            input_ids: `(batch, num_tokens)` token ids for a single field.
            field: one of [`FIELDS`].

        Returns: `(batch, num_tokens, output_dims[field])`.
        """

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):  # pragma: no cover - optional hook
        raise NotImplementedError


@register_text_encoder("lstm")
class LSTMTextEncoder(FGATextEncoder):
    """The encoder from the paper: a shared embedding table and one LSTM per field.

    The five LSTMs are deliberately unshared and differently sized -- questions
    and answers get the wide ones, history and caption the narrow ones -- because
    the concatenated representation feeds a scoring MLP whose width the paper
    tuned.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_embed_dim, padding_idx=0)

        self._output_dims = {
            "question": config.hidden_ques_dim,
            "answer": config.hidden_ans_dim,
            "caption": config.hidden_cap_dim,
            "history_question": config.hidden_hist_dim,
            "history_answer": config.hidden_hist_dim,
        }
        self.lstms = nn.ModuleDict(
            {field: nn.LSTM(config.word_embed_dim, dim, batch_first=True) for field, dim in self._output_dims.items()}
        )

    @property
    def output_dims(self) -> Dict[str, int]:
        return dict(self._output_dims)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.word_embeddings = value

    def forward(self, input_ids: torch.Tensor, field: str) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        sequence, _ = self.lstms[field](embeddings)
        return sequence
