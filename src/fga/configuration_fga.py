"""Configuration class for Factor Graph Attention.

Factor Graph Attention (Schwartz et al., CVPR 2019) -- https://arxiv.org/abs/1904.05880
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import PretrainedConfig


class FGAConfig(PretrainedConfig):
    r"""Configuration for [`FGAModel`].

    The model treats every modality as a *utility*: a set of entities with an
    embedding each (100 answer options, the question words, the caption words,
    the image regions, and the question/answer of each history round). Attention
    over a utility is the softmax of a sum of learned potentials -- unary,
    self-interaction, pairwise interaction with the other utilities, and an
    optional prior -- exactly as in a factor graph.

    Args:
        vocab_size (`int`, *optional*, defaults to 8964):
            Number of rows in the word embedding table. This is
            `len(word2ind) + 3`: the padding id 0, the `len(word2ind)` real
            words, and the `<stop>` / `<empty>` ids appended by the data
            pipeline. Use [`~fga.data.vocab_size_from_params`] to derive it from
            `visdial_params.json`.
        stop_token_id (`int`, *optional*):
            Id of the `<stop>` symbol appended after the last word of a
            question/answer/caption. Defaults to `vocab_size - 2`.
        empty_token_id (`int`, *optional*):
            Id used to fill history rounds that have not happened yet.
            Defaults to `vocab_size - 1`.
        word_embed_dim (`int`, *optional*, defaults to 200):
            Dimension of the shared word embedding table.
        hidden_ques_dim (`int`, *optional*, defaults to 512):
            Hidden size of the question LSTM.
        hidden_ans_dim (`int`, *optional*, defaults to 512):
            Hidden size of the answer-option LSTM.
        hidden_hist_dim (`int`, *optional*, defaults to 128):
            Hidden size of the two history LSTMs (question and answer).
        hidden_cap_dim (`int`, *optional*, defaults to 128):
            Hidden size of the caption LSTM.
        hidden_img_dim (`int`, *optional*, defaults to 2048):
            Dimension of the pre-extracted image region features (2048 for the
            F-RCNN features, 512 for the VGG grid features).
        num_options (`int`, *optional*, defaults to 100):
            Number of candidate answers ranked per dialog round.
        num_history_rounds (`int`, *optional*, defaults to 9):
            Number of previous rounds fed as history (10 rounds - the current one).
        utility_sizes (`List[int]`, *optional*, defaults to `[100, 21, 41, 37, 21, 21]`):
            Number of entities per utility, ordered as
            `[answer, question, caption, image, history-question, history-answer]`.
            These are fixed because the pairwise factors batch-norm over the
            flattened `size_x * size_y` interaction grid. Set `size_force=True` to
            adaptively pool inputs to these sizes instead.
        sharing_factor_weights (`Dict[int, Tuple[int, List[int]]]`, *optional*):
            Maps a utility index to `(num_repeats, connected_utility_indices)`.
            Used so the 9 history rounds share one set of factor weights.
            Defaults to `{4: (9, [0, 1]), 5: (9, [0, 1])}`.

            This index form is the serialized one. To write it readably, pass
            `shared_utilities` instead:

            ```python
            FGAConfig(shared_utilities=[
                {"name": "history_question", "repeats": 9, "connected_to": ["answer", "question"]},
                {"name": "history_answer",   "repeats": 9, "connected_to": ["answer", "question"]},
            ])
            ```

            and read it back with the `shared_utilities` property.
        shared_utilities (`List[Dict]`, *optional*):
            Readable alternative to `sharing_factor_weights`, using the names in
            `fga.modeling_fga.UTILITY_NAMES`. Normalized into
            `sharing_factor_weights` on construction; passing both is an error.
        use_prior (`bool`, *optional*, defaults to `True`):
            Add the length/uniform prior potential.
        use_pairwise (`bool`, *optional*, defaults to `True`):
            Add potentials from interactions between distinct utilities.
        use_unary (`bool`, *optional*, defaults to `True`):
            Add the local (per-entity) potential.
        use_self (`bool`, *optional*, defaults to `True`):
            Add potentials from interactions among entities of the same utility.
        size_force (`bool`, *optional*, defaults to `False`):
            Adaptively average-pool each utility to `utility_sizes` before
            computing factors, rather than requiring inputs to already match.
        unary_dropout (`float`, *optional*, defaults to 0.5):
            Dropout applied inside the unary potential.
        classifier_dropout (`float`, *optional*, defaults to 0.5):
            Dropout applied in the scoring MLP.
        text_encoder_type (`str`, *optional*, defaults to `"lstm"`):
            Which text encoder produces the per-token utility embeddings.
            Only `"lstm"` -- the encoder used in the paper -- ships with this
            repo; the value exists so alternative encoders can be registered
            without changing the checkpoint format.
        legacy_unary_dropout (`bool`, *optional*, defaults to `False`):
            Reproduce a quirk of the original release, where the unary potential
            called `F.dropout` without passing `self.training` and therefore kept
            dropping activations at evaluation time. Set to `True` only to match
            the numbers of the originally released checkpoint exactly.
        initializer_type (`str`, *optional*, defaults to `"he"`):
            Initialization for non-LSTM weights: `"he"`, `"xavier"` or `"default"`.
        lstm_initializer_type (`str`, *optional*, defaults to `"he"`):
            Initialization for LSTM weights: `"he"`, `"xavier"` or `"default"`.
    """

    model_type = "fga"
    attribute_map = {"hidden_size": "hidden_ques_dim"}

    def __init__(
        self,
        vocab_size: int = 8964,
        stop_token_id: Optional[int] = None,
        empty_token_id: Optional[int] = None,
        word_embed_dim: int = 200,
        hidden_ques_dim: int = 512,
        hidden_ans_dim: int = 512,
        hidden_hist_dim: int = 128,
        hidden_cap_dim: int = 128,
        hidden_img_dim: int = 2048,
        num_options: int = 100,
        num_history_rounds: int = 9,
        utility_sizes: Optional[Sequence[int]] = None,
        sharing_factor_weights: Optional[Dict[int, Tuple[int, List[int]]]] = None,
        shared_utilities: Optional[List[Dict[str, Any]]] = None,
        use_prior: bool = True,
        use_pairwise: bool = True,
        use_unary: bool = True,
        use_self: bool = True,
        size_force: bool = False,
        unary_dropout: float = 0.5,
        classifier_dropout: float = 0.5,
        text_encoder_type: str = "lstm",
        legacy_unary_dropout: bool = False,
        initializer_type: str = "he",
        lstm_initializer_type: str = "he",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.stop_token_id = stop_token_id if stop_token_id is not None else vocab_size - 2
        self.empty_token_id = empty_token_id if empty_token_id is not None else vocab_size - 1
        self.word_embed_dim = word_embed_dim
        self.hidden_ques_dim = hidden_ques_dim
        self.hidden_ans_dim = hidden_ans_dim
        self.hidden_hist_dim = hidden_hist_dim
        self.hidden_cap_dim = hidden_cap_dim
        self.hidden_img_dim = hidden_img_dim
        self.num_options = num_options
        self.num_history_rounds = num_history_rounds
        self.utility_sizes = list(utility_sizes) if utility_sizes is not None else [100, 21, 41, 37, 21, 21]
        if shared_utilities is not None:
            if sharing_factor_weights is not None:
                raise ValueError(
                    "Pass either shared_utilities (readable) or sharing_factor_weights (indexed), not both."
                )
            sharing_factor_weights = self._shared_utilities_to_indices(shared_utilities)

        # json round-trips dict keys as strings; normalize back to int.
        if sharing_factor_weights is None:
            sharing_factor_weights = {4: (9, [0, 1]), 5: (9, [0, 1])}
        self.sharing_factor_weights = {
            int(k): (int(v[0]), [int(i) for i in v[1]]) for k, v in sharing_factor_weights.items()
        }
        self.use_prior = use_prior
        self.use_pairwise = use_pairwise
        self.use_unary = use_unary
        self.use_self = use_self
        self.size_force = size_force
        self.unary_dropout = unary_dropout
        self.classifier_dropout = classifier_dropout
        self.text_encoder_type = text_encoder_type
        self.legacy_unary_dropout = legacy_unary_dropout
        self.initializer_type = initializer_type
        self.lstm_initializer_type = lstm_initializer_type

        kwargs.setdefault("pad_token_id", 0)
        super().__init__(**kwargs)

    #: Utility order fixing the meaning of every index in this config.
    UTILITY_NAMES = ("answer", "question", "caption", "image", "history_question", "history_answer")

    @classmethod
    def _shared_utilities_to_indices(cls, shared_utilities: List[Dict[str, Any]]) -> Dict[int, Tuple[int, List[int]]]:
        index_of = {name: i for i, name in enumerate(cls.UTILITY_NAMES)}

        def resolve(name: str) -> int:
            if name not in index_of:
                raise ValueError(f"Unknown utility {name!r}; expected one of {list(cls.UTILITY_NAMES)}.")
            return index_of[name]

        return {
            resolve(entry["name"]): (int(entry["repeats"]), [resolve(n) for n in entry["connected_to"]])
            for entry in shared_utilities
        }

    @property
    def shared_utilities(self) -> List[Dict[str, Any]]:
        """`sharing_factor_weights` rendered with utility names instead of indices."""
        return [
            {
                "name": self.UTILITY_NAMES[index],
                "repeats": repeats,
                "connected_to": [self.UTILITY_NAMES[i] for i in connected],
            }
            for index, (repeats, connected) in sorted(self.sharing_factor_weights.items())
        ]

    @property
    def utility_dims(self) -> List[int]:
        """Embedding dimension of each utility, in canonical utility order."""
        return [
            self.hidden_ans_dim,  # 0: answer options
            self.hidden_ques_dim,  # 1: question
            self.hidden_cap_dim,  # 2: caption
            self.hidden_img_dim,  # 3: image regions
            self.hidden_hist_dim,  # 4: history questions
            self.hidden_hist_dim,  # 5: history answers
        ]
