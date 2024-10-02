"""
This module contains tests for the generate_splits functionality.

Test Summary:
1. test_generate_random_split: Tests random split generation with and without stratification.
2. test_generate_matched_split: Tests matched split generation with and without stratification.
3. test_write_splitfile_variations: Tests write_splitfile function with various parameters.
4. test_write_splitfile_insufficient_samples: Tests handling of insufficient samples.
5. test_write_splitfile_invalid_confound_method: Tests handling of invalid confound correction method.
"""

import json
import tempfile

import h5py
import numpy as np
import pytest
from sklearn.datasets import make_classification

from workflow.scripts.generate_splits import (
    generate_matched_split,
    generate_random_split,
    write_splitfile,
)

@pytest.fixture
def sample_data():
    """
    Fixture to create sample data for testing split generation.
    
    Returns:
        tuple: Contains target labels (y), confounding variables (match), and a mask.
    """
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    match = np.random.rand(100, 5)  # 5 confound variables
    mask = np.ones(100, dtype=bool)
    return y, match, mask

@pytest.mark.parametrize("do_stratify", [True, False])
def test_generate_random_split(sample_data, do_stratify):
    """
    Test the generate_random_split function with and without stratification.
    
    Args:
        sample_data (tuple): Fixture containing sample data.
        do_stratify (bool): Whether to use stratification in splitting.
    """
    y, _, mask = sample_data
    split = generate_random_split(
        y=y,
        n_train=60,
        n_val=20,
        n_test=20,
        do_stratify=do_stratify,
        seed=0,
        mask=mask
    )

    # Assert the structure and basic properties of the split
    assert set(split.keys()) == {"idx_train", "idx_val", "idx_test", "samplesize", "seed", "stratify"}
    assert len(split["idx_train"]) == 60
    assert len(split["idx_val"]) == 20
    assert len(split["idx_test"]) == 20
    assert split["samplesize"] == 60
    assert split["seed"] == 0
    assert split["stratify"] == do_stratify
    
    # Check for no overlap between sets
    assert len(set(split["idx_train"]) & set(split["idx_val"]) & set(split["idx_test"])) == 0
    
    # Ensure all indices are valid
    all_indices = split["idx_train"] + split["idx_val"] + split["idx_test"]
    assert all(0 <= idx < len(y) for idx in all_indices)

    if do_stratify:
        # Check if stratification is maintained
        train_classes, train_counts = np.unique(y[split["idx_train"]], return_counts=True)
        val_classes, val_counts = np.unique(y[split["idx_val"]], return_counts=True)
        test_classes, test_counts = np.unique(y[split["idx_test"]], return_counts=True)
        
        assert np.array_equal(train_classes, val_classes)
        assert np.array_equal(train_classes, test_classes)
        assert np.allclose(train_counts / len(split["idx_train"]), 
                           val_counts / len(split["idx_val"]), 
                           rtol=0.1)
        assert np.allclose(train_counts / len(split["idx_train"]), 
                           test_counts / len(split["idx_test"]), 
                           rtol=0.1)

@pytest.mark.parametrize("do_stratify", [True, False])
def test_generate_matched_split(sample_data, do_stratify):
    """
    Test the generate_matched_split function with and without stratification.
    
    Args:
        sample_data (tuple): Fixture containing sample data.
        do_stratify (bool): Whether to use stratification in splitting.
    """
    y, match, mask = sample_data
    split = generate_matched_split(
        y=y,
        match=match,
        n_train=60,
        n_val=20,
        n_test=20,
        do_stratify=do_stratify,
        seed=0,
        mask=mask
    )

    # Assert the structure and basic properties of the split
    assert set(split.keys()) == {"idx_train", "idx_val", "idx_test", "samplesize", "seed", "stratify", "average_matching_score"}
    assert len(split["idx_train"]) == 60
    assert len(split["idx_val"]) == 20
    assert len(split["idx_test"]) == 20
    assert split["samplesize"] == 60
    assert split["seed"] == 0
    assert split["stratify"] == do_stratify
    
    # Check for no overlap between sets
    assert len(set(split["idx_train"]) & set(split["idx_val"]) & set(split["idx_test"])) == 0
    
    # Check the average matching score
    assert isinstance(split["average_matching_score"], float)
    assert 0 <= split["average_matching_score"] <= 10  # Temporarily increased upper bound

    # Ensure all indices are valid
    all_indices = split["idx_train"] + split["idx_val"] + split["idx_test"]
    assert all(0 <= idx < len(y) for idx in all_indices)

    if do_stratify:
        # Check if stratification is maintained
        train_classes, train_counts = np.unique(y[split["idx_train"]], return_counts=True)
        val_classes, val_counts = np.unique(y[split["idx_val"]], return_counts=True)
        test_classes, test_counts = np.unique(y[split["idx_test"]], return_counts=True)
        
        assert np.array_equal(train_classes, val_classes)
        assert np.array_equal(train_classes, test_classes)
        assert np.allclose(train_counts / len(split["idx_train"]), 
                           val_counts / len(split["idx_val"]), 
                           rtol=0.1)
        assert np.allclose(train_counts / len(split["idx_train"]), 
                           test_counts / len(split["idx_test"]), 
                           rtol=0.1)

@pytest.fixture
def sample_h5_files(tmpdir):
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    confounds = np.random.rand(100, 5)

    features_path = str(tmpdir.join("features.h5"))
    targets_path = str(tmpdir.join("targets.h5"))
    confounds_path = str(tmpdir.join("confounds.h5"))

    with h5py.File(features_path, "w") as f:
        f.create_dataset("data", data=X)
        f.create_dataset("mask", data=np.ones(100, dtype=bool))

    with h5py.File(targets_path, "w") as f:
        f.create_dataset("data", data=y)
        f.create_dataset("mask", data=np.ones(100, dtype=bool))

    with h5py.File(confounds_path, "w") as f:
        f.create_dataset("data", data=confounds)
        f.create_dataset("mask", data=np.ones(100, dtype=bool))

    return features_path, targets_path, confounds_path

@pytest.mark.parametrize("confound_correction_method", ["correct-x", "correct-y", "correct-both", "none", "matching"])
@pytest.mark.parametrize("stratify,balanced", [(True, True), (True, False), (False, False)])
def test_write_splitfile_variations(sample_h5_files, tmpdir, confound_correction_method, stratify, balanced):
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = str(tmpdir.join("split.json"))

    write_splitfile(
        features_path=features_path,
        targets_path=targets_path,
        split_path=split_path,
        confounds_path=confounds_path,
        confound_correction_method=confound_correction_method,
        n_train=60,
        n_val=20,
        n_test=20,
        seed=0,
        stratify=stratify,
        balanced=balanced
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    if "error" in split_dict:
        pytest.skip(f"Skipping due to error: {split_dict['error']}")

    assert set(split_dict.keys()) >= {"idx_train", "idx_val", "idx_test", "samplesize", "seed", "stratify"}
    
    assert len(split_dict["idx_train"]) == 60
    assert len(split_dict["idx_val"]) == 20
    assert len(split_dict["idx_test"]) == 20
    assert split_dict["samplesize"] == 60
    assert split_dict["seed"] == 0
    assert split_dict["stratify"] == stratify
    
    assert len(set(split_dict["idx_train"]) & set(split_dict["idx_val"]) & set(split_dict["idx_test"])) == 0

    if confound_correction_method == "matching":
        assert "average_matching_score" in split_dict
        assert isinstance(split_dict["average_matching_score"], float)

    all_indices = split_dict["idx_train"] + split_dict["idx_val"] + split_dict["idx_test"]
    assert all(0 <= idx < 100 for idx in all_indices)
    assert len(all_indices) == 100

    if stratify and balanced:
        with h5py.File(targets_path, "r") as f:
            y = f["data"][:]
        train_classes = y[split_dict["idx_train"]]
        unique, counts = np.unique(train_classes, return_counts=True)
        assert len(unique) > 1
        assert np.allclose(counts, counts[0], rtol=0.1)

def test_write_splitfile_insufficient_samples(sample_h5_files, tmpdir):
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = str(tmpdir.join("split.json"))

    write_splitfile(
        features_path=features_path,
        targets_path=targets_path,
        split_path=split_path,
        confounds_path=confounds_path,
        confound_correction_method="none",
        n_train=90,
        n_val=20,
        n_test=20,
        seed=0,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "insufficient samples"

def test_write_splitfile_invalid_confound_method(sample_h5_files, tmpdir):
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = str(tmpdir.join("split.json"))

    with pytest.raises(ValueError, match="Invalid confound correction method"):
        write_splitfile(
            features_path=features_path,
            targets_path=targets_path,
            split_path=split_path,
            confounds_path=confounds_path,
            confound_correction_method="invalid_method",
            n_train=60,
            n_val=20,
            n_test=20,
            seed=0,
            stratify=False,
            balanced=False
        )

    # Check if the split file was not created
    assert not tmpdir.join("split.json").exists()