"""
test_fit.py
====================================
This module contains unit tests for the fit function in the fit_model module.
It tests various scenarios including different confound correction methods
and model types (classification and regression).
"""

import json
from typing import Dict, Any, List
import h5py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from workflow.scripts.fit_model import fit


@pytest.mark.parametrize("confound_correction_method", ["with_cni", "only_cni", "normal"])
@pytest.mark.parametrize("model_type", ["classification", "regression"])
def test_fit(tmpdir: str, confound_correction_method: str, model_type: str) -> None:
    """
    Test the fit function with various confound correction methods and model types.

    This test creates toy datasets, sets up the necessary file structure,
    runs the fit function, and checks the output for correctness.

    Args:
        tmpdir (str): Pytest fixture for temporary directory
        confound_correction_method (str): Method for confound correction
        model_type (str): Type of model to test (classification or regression)
    """
    # Create toy problems
    if model_type == "classification":
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        model_name = "ridge-cls"
    else:
        X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        model_name = "ridge-reg"
    
    y = y.reshape(-1, 1)
    cni = np.random.rand(100, 5)

    # Create data split
    split: Dict[str, Any] = {
        "idx_train": list(range(0, 60)),
        "idx_val": list(range(60, 80)),
        "idx_test": list(range(80, 100)),
        "samplesize": 100,
        "seed": 42,
    }

    # Save data to temporary files
    features_path = str(tmpdir.join("features.h5"))
    targets_path = str(tmpdir.join("targets.h5"))
    cni_path = str(tmpdir.join("cni.h5"))
    split_path = str(tmpdir.join("split.json"))

    def create_h5_file(path: str, data: np.ndarray) -> None:
        """Create an HDF5 file with the given data."""
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("mask", data=np.isfinite(data).all(axis=1))

    create_h5_file(features_path, X)
    create_h5_file(targets_path, y)
    create_h5_file(cni_path, cni)

    with open(split_path, "w") as f:
        json.dump(split, f)

    # Define model parameters
    hp = {"alpha": [0.1, 1.0, 10.0]}
    grid = {model_name: hp}

    # Run the fit function
    scores_path = str(tmpdir.join("scores.csv"))
    fit(
        features_path,
        targets_path,
        split_path,
        scores_path,
        model_name,
        grid,
        [],
        confound_correction_method,
        cni_path,
    )

    # Load and check the scores
    scores = pd.read_csv(scores_path)
    print(f"Scores for {model_type} model with {confound_correction_method} method:")
    print(scores)

    # Check expected columns
    expected_columns: List[str] = ["n", "s", "alpha"]
    expected_columns.extend(["acc_train", "acc_val", "acc_test", "f1_train", "f1_val", "f1_test"] if model_type == "classification" 
                            else ["r2_train", "r2_val", "r2_test", "mae_train", "mae_val", "mae_test", "mse_train", "mse_val", "mse_test"])

    for col in expected_columns:
        assert col in scores.columns, f"Expected column {col} not found in scores"

    # Check metadata
    assert scores["n"].iloc[0] == split["samplesize"], f"Expected samplesize {split['samplesize']}, got {scores['n'].iloc[0]}"
    assert scores["s"].iloc[0] == split["seed"], f"Expected seed {split['seed']}, got {scores['s'].iloc[0]}"

    # Check score ranges
    score_columns = [col for col in scores.columns if col.endswith(('train', 'val', 'test'))]
    for col in score_columns:
        if col.startswith(('acc', 'f1')):
            assert all(0 <= scores[col]) and all(scores[col] <= 1), f"Scores in column {col} are not within [0, 1] range"
        elif col.startswith('r2'):
            assert all(-10 <= scores[col]) and all(scores[col] <= 1), f"R² scores in column {col} are outside the expected range"
        else:
            assert all(scores[col] >= 0), f"Scores in column {col} contain negative values"

    # Check alpha values
    assert set(scores["alpha"].astype(float)) == set(hp["alpha"]), f"Expected alpha values {set(hp['alpha'])}, got {set(scores['alpha'].astype(float))}"

    # Additional checks for regression models
    if model_type == "regression":
        r2_cols = ["r2_train", "r2_val", "r2_test"]
        r2_values = scores[r2_cols].values
        r2_range = r2_values.max() - r2_values.min()
        assert r2_range < 0.5, f"R² values vary too much across sets: {dict(zip(r2_cols, r2_values[0]))}"

        mse_cols = ["mse_train", "mse_val", "mse_test"]
        mse_values = scores[mse_cols].values
        assert np.all(mse_values >= 0), f"Negative MSE values found: {dict(zip(mse_cols, mse_values[0]))}"
        mse_range = mse_values.max() - mse_values.min()
        assert mse_range < 1e6, f"MSE values vary too much across sets: {dict(zip(mse_cols, mse_values[0]))}"

    # Print detailed score information
    print("\nDetailed score information:")
    for col in score_columns:
        print(f"{col}: min={scores[col].min():.4f}, max={scores[col].max():.4f}, mean={scores[col].mean():.4f}")


def test_fit_with_existing_scores(tmpdir: str) -> None:
    """
    Test the fit function with existing scores.

    This test checks if the fit function correctly uses existing scores
    and only computes new scores for parameter combinations not already present.

    Args:
        tmpdir (str): Pytest fixture for temporary directory
    """
    # Setup data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    y = y.reshape(-1, 1)
    cni = np.random.rand(100, 5)
    split: Dict[str, Any] = {
        "idx_train": list(range(0, 60)),
        "idx_val": list(range(60, 80)),
        "idx_test": list(range(80, 100)),
        "samplesize": 100,
        "seed": 42,
    }

    # Save data to temporary files
    features_path = str(tmpdir.join("features.h5"))
    targets_path = str(tmpdir.join("targets.h5"))
    cni_path = str(tmpdir.join("cni.h5"))
    split_path = str(tmpdir.join("split.json"))

    def create_h5_file(path: str, data: np.ndarray) -> None:
        """Create an HDF5 file with the given data."""
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("mask", data=np.isfinite(data).all(axis=1))

    create_h5_file(features_path, X)
    create_h5_file(targets_path, y)
    create_h5_file(cni_path, cni)

    with open(split_path, "w") as f:
        json.dump(split, f)

    # Define model parameters
    model_name = "ridge-cls"
    hp = {"alpha": [0.1, 1.0, 10.0]}
    grid = {model_name: hp}

    # Prepare existing scores
    EXISTING_ACC = 0.75
    existing_scores_path = str(tmpdir.join("existing_scores.csv"))
    df_existing_scores = pd.DataFrame([
        {
            "alpha": 0.1,
            "acc_train": EXISTING_ACC,
            "acc_val": EXISTING_ACC,
            "acc_test": EXISTING_ACC,
            "f1_train": EXISTING_ACC,
            "f1_val": EXISTING_ACC,
            "f1_test": EXISTING_ACC,
            "n": 100,
            "s": 42,
        }
    ])
    df_existing_scores.to_csv(existing_scores_path, index=False)

    # Run the fit function
    scores_path = str(tmpdir.join("scores.csv"))
    fit(
        features_path,
        targets_path,
        split_path,
        scores_path,
        model_name,
        grid,
        [existing_scores_path],
        "normal",
        cni_path,
    )

    # Load and check the scores
    scores = pd.read_csv(scores_path)

    # Check if existing score is present and unchanged
    assert EXISTING_ACC in scores["acc_train"].values, "Existing accuracy score not found in results"
    assert len(scores) == 3, f"Expected 3 scores (1 existing + 2 new), but got {len(scores)}"

    # Check if existing score matches input
    existing_score_row = scores[scores["alpha"] == 0.1].iloc[0]
    for metric in ["acc_train", "acc_val", "acc_test", "f1_train", "f1_val", "f1_test"]:
        assert existing_score_row[metric] == EXISTING_ACC, f"Existing score for {metric} does not match expected value"

    # Check if new scores were computed for other alpha values
    assert set(scores["alpha"]) == set(hp["alpha"]), "Not all expected alpha values are present in the results"