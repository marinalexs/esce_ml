"""
test_generate_splits.py
=======================

This module contains tests for the generate_splits functionality in the ESCE workflow.
It verifies the correct behavior of random and matched split generation, as well as
the write_splitfile function under various conditions.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import h5py
import numpy as np
import pytest
from scipy import stats

from workflow.scripts.generate_splits import (
    generate_matched_split,
    generate_random_split,
    write_splitfile,
    MIN_SAMPLES_PER_SET,
    MAX_CLASSES_FOR_STRATIFICATION,
)



# Constants for test parameters
N_SAMPLES = 1000
N_FEATURES = 20
N_TRAIN = 600
N_TRAIN_MATCH = 100
N_VAL = 100
N_TEST = 100
SEED = 0
N_CONFOUNDS = 3


pytestmark = pytest.mark.usefixtures("generate_synth_data")

def assert_valid_split(split: Dict[str, Any], expected_train: int, expected_val: int, expected_test: int):
    """Helper function to assert common split properties."""
    assert set(split.keys()) >= {"idx_train", "idx_val", "idx_test", "samplesize", "seed", "stratify"}
    assert len(split["idx_train"]) == expected_train
    assert len(split["idx_val"]) == expected_val
    assert len(split["idx_test"]) == expected_test
    assert split["samplesize"] == expected_train
    assert split["seed"] == SEED
    
    # Check for no overlap between sets
    assert len(set(split["idx_train"]) & set(split["idx_val"]) & set(split["idx_test"])) == 0
    
    # Ensure all indices are valid
    all_indices = split["idx_train"] + split["idx_val"] + split["idx_test"]
    assert all(0 <= idx < N_SAMPLES for idx in all_indices)

def assert_stratification(y: np.ndarray, split: Dict[str, List[int]]):
    """Helper function to assert stratification is maintained."""
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
def test_generate_random_split(generate_synth_data, do_stratify):
    """Test the generate_random_split function with and without stratification."""
    X, y, _ = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    mask = np.ones(len(y), dtype=bool)
    split = generate_random_split(
        y=y,
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        do_stratify=do_stratify,
        seed=SEED,
        mask=mask
    )

    assert_valid_split(split, N_TRAIN, N_VAL, N_TEST)
    assert split["stratify"] == do_stratify

    if do_stratify:
        assert_stratification(y, split)

@pytest.mark.parametrize("do_stratify", [True, False])
def test_generate_matched_split(generate_synth_data, do_stratify):
    """Test the generate_matched_split function with and without stratification."""
    _, y, match = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    mask = np.ones(len(y), dtype=bool)
    split = generate_matched_split(
        y=y,
        match=match,
        n_train=N_TRAIN_MATCH,
        n_val=N_VAL,
        n_test=N_TEST,
        do_stratify=do_stratify,
        seed=SEED,
        mask=mask
    )

    assert_valid_split(split, N_TRAIN_MATCH, N_VAL, N_TEST)
    assert split["stratify"] == do_stratify
    assert isinstance(split["average_matching_score"], float)

    # Check matching effectiveness using Kolmogorov-Smirnov test
    for set_name, indices in [("train", split["idx_train"]), ("val", split["idx_val"]), ("test", split["idx_test"])]:
        positive_cases = match[indices][y[indices] == 1]
        negative_cases = match[indices][y[indices] == 0]
        
        for i in range(match.shape[1]):
            _, p_value = stats.ks_2samp(positive_cases[:, i], negative_cases[:, i])
            assert p_value > 0.05, f"Distribution mismatch for confound {i} between positive and negative cases in {set_name} set"

@pytest.mark.parametrize("confound_correction_method", ["correct-x", "correct-y", "correct-both", "none", "matching"])
@pytest.mark.parametrize("stratify,balanced", [(True, True), (True, False), (False, False)])
def test_write_splitfile_variations(generate_synth_data, create_dataset, tmpdir, confound_correction_method, stratify, balanced):
    """Test write_splitfile function with various parameters."""
    X, y, confounds = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    dataset = create_dataset(X, y, confounds, 'h5', Path(tmpdir) / "test_data")
    split_path = Path(tmpdir) / "split.json"

    write_splitfile(
        features_path=str(dataset['features']),
        targets_path=str(dataset['targets']),
        split_path=str(split_path),
        confounds_path=str(dataset['confounds']),
        confound_correction_method=confound_correction_method,
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        seed=SEED,
        stratify=stratify,
        balanced=balanced
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    if "error" in split_dict:
        pytest.skip(f"Skipping due to error: {split_dict['error']}")

    assert_valid_split(split_dict, N_TRAIN, N_VAL, N_TEST)
    assert split_dict["stratify"] == (
        confound_correction_method == "matching" or
        (stratify and confound_correction_method not in ['correct-x', 'correct-y', 'correct-both'])
    )
    
    if confound_correction_method == "matching":
        assert "average_matching_score" in split_dict

def test_write_splitfile_insufficient_samples(generate_synth_data, create_dataset, tmpdir):
    """Test write_splitfile function with insufficient samples."""
    X, y, confounds = generate_synth_data(n_samples=N_TRAIN, n_features=N_FEATURES, classification=True, random_state=42)
    dataset = create_dataset(X, y, confounds, 'h5', Path(tmpdir) / "test_data")
    split_path = Path(tmpdir) / "split.json"

    write_splitfile(
        features_path=str(dataset['features']),
        targets_path=str(dataset['targets']),
        split_path=str(split_path),
        confounds_path=str(dataset['confounds']),
        confound_correction_method="none",
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        seed=SEED,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Insufficient samples"

def test_write_splitfile_invalid_confound_method(generate_synth_data, create_dataset, tmpdir):
    """Test write_splitfile function with an invalid confound correction method."""
    X, y, confounds = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    dataset = create_dataset(X, y, confounds, 'h5', Path(tmpdir) / "test_data")
    split_path = Path(tmpdir) / "split.json"

    with pytest.raises(ValueError, match="Invalid confound correction method"):
        write_splitfile(
            features_path=str(dataset['features']),
            targets_path=str(dataset['targets']),
            split_path=str(split_path),
            confounds_path=str(dataset['confounds']),
            confound_correction_method="invalid_method",
            n_train=N_TRAIN,
            n_val=N_VAL,
            n_test=N_TEST,
            seed=SEED,
            stratify=False,
            balanced=False
        )

def test_write_splitfile_regression(generate_synth_data, create_dataset, tmpdir):
    """Test write_splitfile function with regression data."""
    X, y, confounds = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=False, random_state=42)
    dataset = create_dataset(X, y, confounds, 'h5', Path(tmpdir) / "test_regression_data")
    split_path = Path(tmpdir) / "split.json"

    write_splitfile(
        features_path=str(dataset['features']),
        targets_path=str(dataset['targets']),
        split_path=str(split_path),
        confounds_path=str(dataset['confounds']),
        confound_correction_method="correct-x",
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        seed=SEED,
        stratify=True,  # This should be ignored for regression
        balanced=True   # This should be ignored for regression
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert not split_dict["stratify"]
    assert_valid_split(split_dict, N_TRAIN, N_VAL, N_TEST)

def test_write_splitfile_small_validation_test(generate_synth_data, create_dataset, tmpdir):
    """Test write_splitfile function with small validation and test sets."""
    X, y, confounds = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    dataset = create_dataset(X, y, confounds, 'h5', Path(tmpdir) / "test_data")
    split_path = Path(tmpdir) / "split.json"

    write_splitfile(
        features_path=str(dataset['features']),
        targets_path=str(dataset['targets']),
        split_path=str(split_path),
        confounds_path=str(dataset['confounds']),
        confound_correction_method="none",
        n_train=N_TRAIN,
        n_val=MIN_SAMPLES_PER_SET - 1,
        n_test=MIN_SAMPLES_PER_SET - 1,
        seed=SEED,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Insufficient samples"

def test_write_splitfile_single_class(generate_synth_data, create_dataset, tmpdir):
    """Test write_splitfile function with a single class in the target."""
    X, y, confounds = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    dataset = create_dataset(X, y, confounds, 'h5', Path(tmpdir) / "test_data")
    split_path = Path(tmpdir) / "split.json"

    # Modify targets to have a single class
    with h5py.File(dataset['targets'], "r+") as f:
        f["data"][:] = 0

    write_splitfile(
        features_path=str(dataset['features']),
        targets_path=str(dataset['targets']),
        split_path=str(split_path),
        confounds_path=str(dataset['confounds']),
        confound_correction_method="none",
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        seed=SEED,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Only a single class in target"

def test_write_splitfile_matching_non_binary(generate_synth_data, create_dataset, tmpdir):
    """Test write_splitfile function with matching for non-binary classification."""
    X, y, confounds = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    dataset = create_dataset(X, y, confounds, 'h5', Path(tmpdir) / "test_data")
    split_path = Path(tmpdir) / "split.json"

    # Modify targets to have more than two classes
    with h5py.File(dataset['targets'], "r+") as f:
        f["data"][:] = np.random.randint(0, 3, size=N_SAMPLES)

    write_splitfile(
        features_path=str(dataset['features']),
        targets_path=str(dataset['targets']),
        split_path=str(split_path),
        confounds_path=str(dataset['confounds']),
        confound_correction_method="matching",
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        seed=SEED,
        stratify=False,
        balanced=False
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    assert "error" in split_dict
    assert split_dict["error"] == "Matching requires binary classification"

def test_generate_matched_split_with_biased_confounds(generate_synth_data):
    """
    Test the generate_matched_split function with a dataset where cases and controls
    have clearly different confound distributions.
    """
    
    X, y, _ = generate_synth_data(n_samples=N_SAMPLES, n_features=N_FEATURES, classification=True, random_state=42)
    
    # Create biased confounds
    confounds_cases = np.random.normal(loc=0.5, scale=1, size=(N_SAMPLES // 2, N_CONFOUNDS))
    confounds_controls = np.random.normal(loc=-0.5, scale=1, size=(N_SAMPLES // 2, N_CONFOUNDS))
    confounds = np.vstack([confounds_cases, confounds_controls])
    
    split = generate_matched_split(
        y=y,
        match=confounds,
        n_train=N_TRAIN_MATCH,
        n_val=N_VAL,
        n_test=N_TEST,
        do_stratify=True,
        seed=SEED,
        mask=None
    )
    
    assert_valid_split(split, N_TRAIN_MATCH, N_VAL, N_TEST)
    
    def calculate_ks_stats(group1: np.ndarray, group2: np.ndarray) -> tuple:
        """Calculate KS statistic and p-value for each confound."""
        ks_stats = []
        p_values = []
        for i in range(N_CONFOUNDS):
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
        
        assert split_ks < original_ks, f"KS statistic for {split_name} should be smaller than original"
        assert split_p > original_p, f"p-value for {split_name} should be larger than original"
        assert split_p > 0.05, f"p-value for {split_name} should be non-significant"

    assert 0 < split["average_matching_score"] < 1, "Average matching score should be between 0 and 1"