"""Tests for assembling extracted features into the h5 the model reads.

The ordering guarantee is the one that matters: row `i` must be the image at
position `i` of `unique_img_{split}`, because FGA looks images up by position.
A mismatch here pairs every dialog with the wrong picture and still trains.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_features_h5.py"

IMAGES = [
    "VisualDialog_val2018/VisualDialog_val2018_000000000001.jpg",
    "VisualDialog_val2018/VisualDialog_val2018_000000000002.jpg",
    "VisualDialog_val2018/VisualDialog_val2018_000000000003.jpg",
]


@pytest.fixture
def workspace(tmp_path):
    params = {"word2ind": {"a": 1}, "unique_img_val": IMAGES}
    params_path = tmp_path / "visdial_params.json"
    params_path.write_text(json.dumps(params))

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    # Each image gets a constant-valued block, so a misordering is detectable.
    for i, name in enumerate(IMAGES):
        stem = os.path.splitext(os.path.basename(name))[0]
        np.save(features_dir / f"{stem}.npy", np.full((36, 8), float(i + 1), dtype=np.float32))
        # The extractor also writes an _info sidecar, which must be ignored.
        np.save(features_dir / f"{stem}_info.npy", np.zeros((36, 4), dtype=np.float32))

    return {
        "params": params_path,
        "features_dir": features_dir,
        "output": tmp_path / "frcnn_features_new.h5",
    }


def run_builder(workspace, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--features_dir",
            str(workspace["features_dir"]),
            "--split",
            "val",
            "--visdial_params_path",
            str(workspace["params"]),
            "--output",
            str(workspace["output"]),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_rows_follow_the_params_image_order(workspace):
    result = run_builder(workspace)
    assert result.returncode == 0, result.stderr

    with h5py.File(workspace["output"], "r") as h5:
        features = h5["val_features"][:]

    assert features.shape == (3, 37, 8)
    for i in range(3):
        # Row i must hold image i's constant, on the 36 real proposals.
        assert np.allclose(features[i, 1:], float(i + 1)), f"row {i} holds the wrong image"


def test_global_region_is_the_mean_of_the_proposals(workspace):
    run_builder(workspace)
    with h5py.File(workspace["output"], "r") as h5:
        features = h5["val_features"][:]
    for i in range(3):
        np.testing.assert_allclose(features[i, 0], features[i, 1:].mean(axis=0), rtol=1e-5)


def test_no_global_region_flag_keeps_only_proposals(workspace):
    run_builder(workspace, "--no_global_region", "--num_regions", "36")
    with h5py.File(workspace["output"], "r") as h5:
        features = h5["val_features"][:]
    assert features.shape == (3, 36, 8)
    assert np.allclose(features[0], 1.0)


def test_missing_features_are_refused_by_default(workspace):
    stem = os.path.splitext(os.path.basename(IMAGES[1]))[0]
    (workspace["features_dir"] / f"{stem}.npy").unlink()

    result = run_builder(workspace)
    assert result.returncode != 0
    assert "Refusing to write a features file with gaps" in result.stdout + result.stderr
    assert not workspace["output"].exists()


def test_missing_features_can_be_zero_filled_deliberately(workspace):
    stem = os.path.splitext(os.path.basename(IMAGES[1]))[0]
    (workspace["features_dir"] / f"{stem}.npy").unlink()

    result = run_builder(workspace, "--allow_missing")
    assert result.returncode == 0, result.stderr

    with h5py.File(workspace["output"], "r") as h5:
        features = h5["val_features"][:]
    assert np.allclose(features[1], 0.0)
    # The surrounding rows keep their identity rather than shifting up.
    assert np.allclose(features[0, 1:], 1.0)
    assert np.allclose(features[2, 1:], 3.0)


def test_fewer_proposals_than_regions_are_padded(workspace, tmp_path):
    stem = os.path.splitext(os.path.basename(IMAGES[0]))[0]
    np.save(workspace["features_dir"] / f"{stem}.npy", np.ones((10, 8), dtype=np.float32))

    run_builder(workspace)
    with h5py.File(workspace["output"], "r") as h5:
        row = h5["val_features"][0]
    assert np.allclose(row[1:11], 1.0)
    assert np.allclose(row[11:], 0.0)


def test_existing_split_is_not_clobbered_without_overwrite(workspace):
    assert run_builder(workspace).returncode == 0
    second = run_builder(workspace)
    assert second.returncode != 0
    assert "--overwrite" in second.stdout + second.stderr


def test_output_is_readable_by_the_dataset_loader(workspace):
    run_builder(workspace)
    with h5py.File(workspace["output"], "r") as h5:
        assert "val_features" in h5
        assert h5["val_features"].dtype == np.float32
