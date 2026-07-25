"""Data-pipeline tests.

The dialog tests run against the real `data/visdial_data.h5` when it is present
and skip otherwise, so the suite still passes on a fresh clone.
"""

import os

import h5py
import numpy as np
import pytest
import torch

from fga.data import FEATURE_KEYS, VisDialCollator, VisDialDataset, load_visdial_params, vocab_size_from_params

DATA_H5 = "data/visdial_data.h5"
PARAMS_JSON = "data/visdial_params.json"

needs_real_data = pytest.mark.skipif(
    not (os.path.exists(DATA_H5) and os.path.exists(PARAMS_JSON)),
    reason="data/visdial_data.h5 and data/visdial_params.json are required",
)


@pytest.fixture(scope="module")
def dev_features(tmp_path_factory):
    """A tiny random stand-in for the F-RCNN feature dump."""
    path = tmp_path_factory.mktemp("features") / "dev_features.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5:
        for split in ("train", "val", "test"):
            h5.create_dataset(f"{split}_features", data=rng.standard_normal((8, 37, 16), dtype=np.float32))
    return str(path)


@pytest.fixture(scope="module")
def val_dataset(dev_features):
    if not os.path.exists(DATA_H5):
        pytest.skip("no visdial_data.h5")
    vocab_size = vocab_size_from_params(load_visdial_params(PARAMS_JSON))
    return VisDialDataset(
        visdial_data_path=DATA_H5,
        image_features_path=dev_features,
        split="val",
        vocab_size=vocab_size,
        limit_images=8,
    )


@needs_real_data
def test_vocab_size_covers_the_stop_and_empty_symbols():
    params = load_visdial_params(PARAMS_JSON)
    vocab_size = vocab_size_from_params(params)
    # ids run 0 (pad), 1..N (words), N+1 (stop), N+2 (empty)
    assert vocab_size == len(params["word2ind"]) + 3
    assert max(params["word2ind"].values()) == len(params["word2ind"])


@needs_real_data
def test_example_has_the_shapes_the_model_declares(val_dataset):
    example = val_dataset[0]
    assert set(example) == set(FEATURE_KEYS) | {"labels"}
    assert example["question_input_ids"].shape == (21,)
    assert example["option_input_ids"].shape == (100, 21)
    assert example["history_question_input_ids"].shape == (9, 21)
    assert example["history_answer_input_ids"].shape == (9, 21)
    assert example["caption_input_ids"].shape == (41,)
    assert example["option_lengths"].shape == (100,)
    assert example["image_features"].shape == (37, 16)


@needs_real_data
def test_token_ids_stay_inside_the_embedding_table(val_dataset):
    vocab_size = vocab_size_from_params(load_visdial_params(PARAMS_JSON))
    for index in (0, 1, 5, len(val_dataset) - 1):
        example = val_dataset[index]
        for key in ("question_input_ids", "option_input_ids", "caption_input_ids"):
            assert example[key].max() < vocab_size, key
            assert example[key].min() >= 0, key


@needs_real_data
def test_stop_symbol_lands_on_the_last_real_token(val_dataset):
    example = val_dataset[0]
    assert example["question_input_ids"][example["question_lengths"] - 1] == val_dataset.stop_id
    assert example["caption_input_ids"][example["caption_lengths"] - 1] == val_dataset.stop_id
    lengths = example["option_lengths"]
    ids = example["option_input_ids"]
    assert np.all(ids[np.arange(len(lengths)), lengths - 1] == val_dataset.stop_id)


@needs_real_data
def test_first_round_has_entirely_empty_history(val_dataset):
    first_round = val_dataset[0]
    assert np.all(first_round["history_question_input_ids"][:, 0] == val_dataset.empty_id)
    assert np.all(first_round["history_question_input_ids"][:, 1] == val_dataset.stop_id)


@needs_real_data
def test_history_grows_by_one_round_each_turn(val_dataset):
    """Round k must carry exactly k filled history rows."""
    for round_index in (0, 1, 4, 9):
        example = val_dataset[round_index]
        filled = (example["history_question_input_ids"][:, 0] != val_dataset.empty_id).sum()
        assert filled == round_index


@needs_real_data
def test_option_lengths_index_the_option_pool_not_the_round(val_dataset):
    """Regression: the no-stop branch indexed opt_length_list by round index."""
    dataset = VisDialDataset(
        visdial_data_path=DATA_H5,
        image_features_path=val_dataset.image_features_path,
        split="val",
        vocab_size=vocab_size_from_params(load_visdial_params(PARAMS_JSON)),
        limit_images=8,
        add_stop_to_answers=False,
        add_stop_to_questions=False,
    )
    example = dataset[3]
    ids, lengths = example["option_input_ids"], example["option_lengths"]
    # The token just before the padding must be a real word for every option.
    assert np.all(ids[np.arange(100), lengths - 1] != 0)


@needs_real_data
def test_labels_point_at_a_real_option(val_dataset):
    for index in (0, 3, 11):
        example = val_dataset[index]
        assert 0 <= example["labels"] < 100


@needs_real_data
def test_image_features_are_l2_normalized(val_dataset):
    example = val_dataset[0]
    norm = np.linalg.norm(example["image_features"].reshape(-1))
    assert np.isclose(norm, 1.0, atol=1e-5)


@needs_real_data
def test_streaming_and_in_memory_agree(dev_features):
    kwargs = dict(
        visdial_data_path=DATA_H5,
        image_features_path=dev_features,
        split="val",
        vocab_size=vocab_size_from_params(load_visdial_params(PARAMS_JSON)),
        limit_images=4,
    )
    eager = VisDialDataset(**kwargs, in_memory=True)
    lazy = VisDialDataset(**kwargs, in_memory=False)
    np.testing.assert_allclose(eager[7]["image_features"], lazy[7]["image_features"], atol=1e-6)


@needs_real_data
def test_collator_builds_a_batch_the_model_accepts(val_dataset):
    batch = VisDialCollator()([val_dataset[i] for i in range(4)])
    assert batch["question_input_ids"].shape == (4, 21)
    assert batch["option_input_ids"].shape == (4, 100, 21)
    assert batch["image_features"].dtype == torch.float32
    assert batch["labels"].shape == (4,)


@needs_real_data
def test_test_split_has_no_labels(dev_features):
    dataset = VisDialDataset(
        visdial_data_path=DATA_H5,
        image_features_path=dev_features,
        split="test",
        vocab_size=vocab_size_from_params(load_visdial_params(PARAMS_JSON)),
        limit_images=4,
    )
    assert "labels" not in dataset[0]
    assert len(dataset.num_rounds_per_image) == 4


def test_missing_feature_key_names_the_available_ones(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("images_train", data=np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(KeyError, match="images_train"):
        VisDialDataset(
            visdial_data_path=DATA_H5 if os.path.exists(DATA_H5) else str(path),
            image_features_path=str(path),
            split="train",
            vocab_size=100,
        )


@needs_real_data
def test_build_hf_dataset_produces_a_datasets_dataset(dev_features, tmp_path):
    """The optional `datasets` path advertised in the README."""
    datasets = pytest.importorskip("datasets")

    dataset = VisDialDataset(
        visdial_data_path=DATA_H5,
        image_features_path=dev_features,
        split="val",
        vocab_size=vocab_size_from_params(load_visdial_params(PARAMS_JSON)),
        limit_images=1,
    )
    from fga.data import build_hf_dataset

    hf_dataset = build_hf_dataset(dataset, cache_dir=str(tmp_path))
    assert isinstance(hf_dataset, datasets.Dataset)
    assert len(hf_dataset) == len(dataset)
    assert set(FEATURE_KEYS).issubset(hf_dataset.column_names)

    batch = VisDialCollator()([hf_dataset[i] for i in range(2)])
    assert batch["option_input_ids"].shape == (2, 100, 21)


@needs_real_data
def test_too_few_image_features_is_reported_clearly(dev_features):
    """The features file must cover the split; running off the end used to be an IndexError."""
    with pytest.raises(ValueError, match="dialogs"):
        VisDialDataset(
            visdial_data_path=DATA_H5,
            image_features_path=dev_features,  # only 8 images
            split="val",  # 2064 dialogs
            vocab_size=vocab_size_from_params(load_visdial_params(PARAMS_JSON)),
        )
