"""Configuration class for Factor Graph Attention.

Factor Graph Attention (Schwartz et al., CVPR 2019) -- https://arxiv.org/abs/1904.05880
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import PretrainedConfig


class FGAConfig(PretrainedConfig):
    r"""Configuration for [`FGAModel`].

    Every modality is a set of entities with an embedding each: the 100 answer
    options, the question words, the caption words, the image regions, and the
    question/answer of each history round. Attention over a modality is the
    softmax of a sum of learned potentials -- unary, self-interaction, pairwise
    interaction with the other modalities, and an optional prior -- exactly as in
    a factor graph.

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
            Number of entities per modality, ordered as
            `[answer, question, caption, image, history-question, history-answer]`.
            These are fixed because the pairwise factors batch-norm over the
            flattened `size_x * size_y` interaction grid. Set `size_force=True` to
            adaptively pool inputs to these sizes instead.
        sharing_factor_weights (`Dict[int, Tuple[int, List[int]]]`, *optional*):
            Maps a modality index to `(num_repeats, connected_modality_indices)`.
            Used so the 9 history rounds share one set of factor weights.
            Defaults to `{4: (9, [0, 1]), 5: (9, [0, 1])}`.

            This index form is the serialized one. To write it readably, pass
            `share_weights` instead.
        share_weights (`List[Dict]`, *optional*):
            Readable alternative to `sharing_factor_weights`, naming the
            modalities that share one set of factor weights, in the same spelling
            the attention layer uses:

            ```python
            FGAConfig(share_weights=[
                {"modalities": [f"history_question_{i}" for i in range(1, 10)],
                 "connected_to": ["answer", "question"]},
                {"modalities": [f"history_answer_{i}" for i in range(1, 10)],
                 "connected_to": ["answer", "question"]},
            ])
            ```

            The connections live on the group rather than on each member, since
            members sharing weights must agree on them. Normalized into
            `sharing_factor_weights` on construction; passing both is an error.
            Read it back with the `share_weights` property.
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
        share_weights: Optional[List[Dict[str, Any]]] = None,
        use_prior: bool = True,
        use_pairwise: bool = True,
        use_unary: bool = True,
        use_self: bool = True,
        size_force: bool = False,
        unary_dropout: float = 0.5,
        classifier_dropout: float = 0.5,
        text_encoder_type: str = "lstm",
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
        if share_weights is not None:
            if sharing_factor_weights is not None:
                raise ValueError("Pass either share_weights (readable) or sharing_factor_weights (indexed), not both.")
            sharing_factor_weights = self._share_weights_to_indices(share_weights)

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
        self.initializer_type = initializer_type
        self.lstm_initializer_type = lstm_initializer_type

        kwargs.setdefault("pad_token_id", 0)
        super().__init__(**kwargs)

    #: Modality order fixing the meaning of every index in this config.
    MODALITY_NAMES = ("answer", "question", "caption", "image", "history_question", "history_answer")

    #: The paper's term for the same ordering.
    UTILITY_NAMES = MODALITY_NAMES

    @classmethod
    def _share_weights_to_indices(cls, share_weights: List[Dict[str, Any]]) -> Dict[int, Tuple[int, List[int]]]:
        """Collapse named groups back into the indexed form that is serialized."""
        index_of = {name: i for i, name in enumerate(cls.MODALITY_NAMES)}

        def resolve(name: str) -> int:
            if name not in index_of:
                raise ValueError(f"Unknown modality {name!r}; expected one of {list(cls.MODALITY_NAMES)}.")
            return index_of[name]

        sharing: Dict[int, Tuple[int, List[int]]] = {}
        for group in share_weights:
            members = list(group["modalities"])
            if len(members) < 2:
                raise ValueError(f"A share_weights group needs at least two modalities; got {members}.")

            # Members are the repeats of one canonical modality: history_question_3
            # belongs to history_question.
            bases = {cls._base_name(name) for name in members}
            if len(bases) != 1:
                raise ValueError(f"A share_weights group must cover one modality; {members} spans {sorted(bases)}.")

            base = bases.pop()
            sharing[resolve(base)] = (len(members), [resolve(n) for n in group.get("connected_to", ())])
        return sharing

    @staticmethod
    def _base_name(name: str) -> str:
        """`history_question_3` -> `history_question`."""
        head, _, tail = name.rpartition("_")
        return head if head and tail.isdigit() else name

    @property
    def share_weights(self) -> List[Dict[str, Any]]:
        """`sharing_factor_weights` rendered as named groups.

        Each repeated modality is expanded into its individual copies, so this
        reads the way the attention layer's `share_weights` argument is written.
        """
        return [
            {
                "modalities": [f"{self.MODALITY_NAMES[index]}_{r + 1}" for r in range(repeats)],
                "connected_to": [self.MODALITY_NAMES[i] for i in connected],
            }
            for index, (repeats, connected) in sorted(self.sharing_factor_weights.items())
        ]

    @property
    def modality_dims(self) -> List[int]:
        """Embedding dimension of each modality, in canonical order."""
        return [
            self.hidden_ans_dim,  # 0: answer options
            self.hidden_ques_dim,  # 1: question
            self.hidden_cap_dim,  # 2: caption
            self.hidden_img_dim,  # 3: image regions
            self.hidden_hist_dim,  # 4: history questions
            self.hidden_hist_dim,  # 5: history answers
        ]

    @property
    def utility_dims(self) -> List[int]:
        """The paper's name for [`modality_dims`]."""
        return self.modality_dims
