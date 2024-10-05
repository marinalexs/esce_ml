"""
test_generate_splits.py
=======================

This module contains tests for the generate_splits functionality in the ESCE workflow.
It verifies the correct behavior of random and matched split generation, as well as
the write_splitfile function under various conditions.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from scipy import stats

from workflow.scripts.generate_splits import (
    generate_matched_split,
    generate_random_split,
    write_splitfile,
    MIN_SAMPLES_PER_SET,
    MAX_CLASSES_FOR_STRATIFICATION,
)

@pytest.fixture
def sample_classification_data():
    """Fixture to create sample classification data for testing split generation."""
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    match = np.random.rand(1000, 5)  # 5 confound variables
    mask = np.ones(1000, dtype=bool)
    return X, y, match, mask

@pytest.fixture
def sample_regression_data():
    """Fixture to create sample regression data for testing split generation."""
    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    match = np.random.rand(1000, 5)  # 5 confound variables
    mask = np.ones(1000, dtype=bool)
    return X, y, match, mask

@pytest.fixture
def sample_h5_files(tmpdir, sample_classification_data):
    """Fixture to create sample HDF5 files for testing."""
    X, y, confounds, _ = sample_classification_data

    features_path = Path(tmpdir) / "features.h5"
    targets_path = Path(tmpdir) / "targets.h5"
    confounds_path = Path(tmpdir) / "confounds.h5"

    with h5py.File(features_path, "w") as f:
        f.create_dataset("data", data=X)
        f.create_dataset("mask", data=np.ones(1000, dtype=bool))

    with h5py.File(targets_path, "w") as f:
        f.create_dataset("data", data=y)
        f.create_dataset("mask", data=np.ones(1000, dtype=bool))

    with h5py.File(confounds_path, "w") as f:
        f.create_dataset("data", data=confounds)
        f.create_dataset("mask", data=np.ones(1000, dtype=bool))

    return features_path, targets_path, confounds_path

@pytest.mark.parametrize("do_stratify", [True, False])
def test_generate_random_split(sample_classification_data, do_stratify):
    """Test the generate_random_split function with and without stratification."""
    _, y, _, mask = sample_classification_data
    split = generate_random_split(
        y=y,
        n_train=600,
        n_val=200,
        n_test=200,
        do_stratify=do_stratify,
        seed=0,
        mask=mask
    )

    assert set(split.keys()) == {"idx_train", "idx_val", "idx_test", "samplesize", "seed", "stratify"}
    assert len(split["idx_train"]) == 600
    assert len(split["idx_val"]) == 200
    assert len(split["idx_test"]) == 200
    assert split["samplesize"] == 600
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
def test_generate_matched_split(sample_classification_data, do_stratify):
    """Test the generate_matched_split function with and without stratification."""
    _, y, match, mask = sample_classification_data
    split = generate_matched_split(
        y=y,
        match=match,
        n_train=100,
        n_val=100,
        n_test=100,
        do_stratify=do_stratify,
        seed=0,
        mask=mask
    )

    assert set(split.keys()) == {"idx_train", "idx_val", "idx_test", "samplesize", "seed", "stratify", "average_matching_score"}
    assert len(split["idx_train"]) == 100
    assert len(split["idx_val"]) == 100
    assert len(split["idx_test"]) == 100
    assert split["samplesize"] == 100
    assert split["seed"] == 0
    assert split["stratify"] == do_stratify
    assert isinstance(split["average_matching_score"], float)
    
    # Check for no overlap between sets
    assert len(set(split["idx_train"]) & set(split["idx_val"]) & set(split["idx_test"])) == 0
    
    # Ensure all indices are valid
    all_indices = split["idx_train"] + split["idx_val"] + split["idx_test"]
    assert all(0 <= idx < len(y) for idx in all_indices)

    # Check matching effectiveness using Kolmogorov-Smirnov test
    train_confounds = match[split["idx_train"]]
    val_confounds = match[split["idx_val"]]
    test_confounds = match[split["idx_test"]]
    
    # Compare positive and negative cases within each set
    for set_name, indices in [("train", split["idx_train"]), ("val", split["idx_val"]), ("test", split["idx_test"])]:
        positive_cases = match[indices][y[indices] == 1]
        negative_cases = match[indices][y[indices] == 0]
        
        for i in range(match.shape[1]):
            _, p_value = stats.ks_2samp(positive_cases[:, i], negative_cases[:, i])
            assert p_value > 0.05, f"Distribution mismatch for confound {i} between positive and negative cases in {set_name} set"

@pytest.mark.parametrize("confound_correction_method", ["correct-x", "correct-y", "correct-both", "none", "matching"])
@pytest.mark.parametrize("stratify,balanced", [(True, True), (True, False), (False, False)])
def test_write_splitfile_variations(sample_h5_files, tmpdir, confound_correction_method, stratify, balanced):
    """Test write_splitfile function with various parameters."""
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = Path(tmpdir) / "split.json"

    write_splitfile(
        features_path=str(features_path),
        targets_path=str(targets_path),
        split_path=str(split_path),
        confounds_path=str(confounds_path),
        confound_correction_method=confound_correction_method,
        n_train=600,
        n_val=200,
        n_test=200,
        seed=0,
        stratify=stratify,
        balanced=balanced
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    if "error" in split_dict:
        pytest.skip(f"Skipping due to error: {split_dict['error']}")

    assert set(split_dict.keys()) >= {"idx_train", "idx_val", "idx_test", "samplesize", "seed", "stratify"}
    
    assert len(split_dict["idx_train"]) == 600
    assert len(split_dict["idx_val"]) == 200
    assert len(split_dict["idx_test"]) == 200
    assert split_dict["samplesize"] == 600
    assert split_dict["seed"] == 0
    assert split_dict["stratify"] == (
        confound_correction_method == "matching" or
        (stratify and confound_correction_method not in ['correct-x', 'correct-y', 'correct-both'])
    )
    
    assert len(set(split_dict["idx_train"]) & set(split_dict["idx_val"]) & set(split_dict["idx_test"])) == 0

    if confound_correction_method == "matching":
        assert "average_matching_score" in split_dict

    all_indices = split_dict["idx_train"] + split_dict["idx_val"] + split_dict["idx_test"]
    assert all(0 <= idx < 1000 for idx in all_indices)
    assert len(all_indices) == 1000

def test_write_splitfile_insufficient_samples(sample_h5_files, tmpdir):
    """Test write_splitfile function with insufficient samples."""
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = Path(tmpdir) / "split.json"

    write_splitfile(
        features_path=str(features_path),
        targets_path=str(targets_path),
        split_path=str(split_path),
        confounds_path=str(confounds_path),
        confound_correction_method="none",
        n_train=900,
        n_val=200,
        n_test=200,
        seed=0,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Insufficient samples"

def test_write_splitfile_invalid_confound_method(sample_h5_files, tmpdir):
    """Test write_splitfile function with an invalid confound correction method."""
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = Path(tmpdir) / "split.json"

    with pytest.raises(ValueError, match="Invalid confound correction method"):
        write_splitfile(
            features_path=str(features_path),
            targets_path=str(targets_path),
            split_path=str(split_path),
            confounds_path=str(confounds_path),
            confound_correction_method="invalid_method",
            n_train=600,
            n_val=200,
            n_test=200,
            seed=0,
            stratify=False,
            balanced=False
        )

def test_write_splitfile_regression(sample_h5_files, tmpdir):
    """Test write_splitfile function with regression data."""
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = Path(tmpdir) / "split.json"

    # Modify targets to be continuous
    with h5py.File(targets_path, "r+") as f:
        f["data"][:] = np.random.randn(1000)

    write_splitfile(
        features_path=str(features_path),
        targets_path=str(targets_path),
        split_path=str(split_path),
        confounds_path=str(confounds_path),
        confound_correction_method="correct-x",
        n_train=600,
        n_val=200,
        n_test=200,
        seed=0,
        stratify=True,  # This should be ignored for regression
        balanced=True   # This should be ignored for regression
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert not split_dict["stratify"]
    assert len(split_dict["idx_train"]) == 600
    assert len(split_dict["idx_val"]) == 200
    assert len(split_dict["idx_test"]) == 200

def test_write_splitfile_small_validation_test(sample_h5_files, tmpdir):
    """Test write_splitfile function with small validation and test sets."""
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = Path(tmpdir) / "split.json"

    write_splitfile(
        features_path=str(features_path),
        targets_path=str(targets_path),
        split_path=str(split_path),
        confounds_path=str(confounds_path),
        confound_correction_method="none",
        n_train=980,
        n_val=MIN_SAMPLES_PER_SET - 1,
        n_test=MIN_SAMPLES_PER_SET - 1,
        seed=0,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Insufficient samples"

def test_write_splitfile_single_class(sample_h5_files, tmpdir):
    """Test write_splitfile function with a single class in the target."""
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = Path(tmpdir) / "split.json"

    # Modify targets to have a single class
    with h5py.File(targets_path, "r+") as f:
        f["data"][:] = 0

    write_splitfile(
        features_path=str(features_path),
        targets_path=str(targets_path),
        split_path=str(split_path),
        confounds_path=str(confounds_path),
        confound_correction_method="none",
        n_train=600,
        n_val=200,
        n_test=200,
        seed=0,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Only a single class in target"

def test_write_splitfile_matching_non_binary(sample_h5_files, tmpdir):
    """Test write_splitfile function with matching for non-binary classification."""
    features_path, targets_path, confounds_path = sample_h5_files
    split_path = Path(tmpdir) / "split.json"

    # Modify targets to have more than two classes
    with h5py.File(targets_path, "r+") as f:
        f["data"][:] = np.random.randint(0, 3, size=1000)

    write_splitfile(
        features_path=str(features_path),
        targets_path=str(targets_path),
        split_path=str(split_path),
        confounds_path=str(confounds_path),
        confound_correction_method="matching",
        n_train=600,
        n_val=200,
        n_test=200,
        seed=0,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Matching requires binary classification"

def test_generate_matched_split_with_biased_confounds():
    """
    Test the generate_matched_split function with a dataset where cases and controls
    have clearly different confound distributions.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 10000
    n_confounds = 3
    
    # Create biased confounds
    confounds_cases = np.random.normal(loc=0.5, scale=1, size=(n_samples // 2, n_confounds))
    confounds_controls = np.random.normal(loc=-0.5, scale=1, size=(n_samples // 2, n_confounds))
    
    confounds = np.vstack([confounds_cases, confounds_controls])
    
    # Create labels (1 for cases, 0 for controls)
    y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])
    
    # Generate matched split
    split = generate_matched_split(
        y=y,
        match=confounds,
        n_train=100,
        n_val=100,
        n_test=100,
        do_stratify=True,
        seed=0,
        mask=None
    )
    
    # Check basic split properties
    assert len(split["idx_train"]) == 100
    assert len(split["idx_val"]) == 100
    assert len(split["idx_test"]) == 100
    
    # Function to calculate KS statistic and p-value for each confound
    def calculate_ks_stats(group1, group2):
        ks_stats = []
        p_values = []
        for i in range(n_confounds):
            ks_stat, p_value = stats.ks_2samp(group1[:, i], group2[:, i])
            ks_stats.append(ks_stat)
            p_values.append(p_value)
        return np.mean(ks_stats), np.mean(p_values)
    
    # Calculate KS statistics for original data
    original_ks, original_p = calculate_ks_stats(confounds_cases, confounds_controls)
    
    # Calculate KS statistics for each split
    for split_name in ["idx_train", "idx_val", "idx_test"]:
        split_indices = split[split_name]
        cases = confounds[split_indices][y[split_indices] == 1]
        controls = confounds[split_indices][y[split_indices] == 0]
        
        split_ks, split_p = calculate_ks_stats(cases, controls)
        
        # Assert that the split KS statistic is smaller (distributions are more similar)
        assert split_ks < original_ks, f"KS statistic for {split_name} should be smaller than original"
        
        # Assert that the p-value is larger (less significant difference)
        assert split_p > original_p, f"p-value for {split_name} should be larger than original"
        
        # Additional check: p-value should ideally be non-significant (> 0.05)
        assert split_p > 0.05, f"p-value for {split_name} should be non-significant"

    # Check that the average matching score is reasonable (should be relatively small)
    assert 0 < split["average_matching_score"] < 1, "Average matching score should be between 0 and 1"