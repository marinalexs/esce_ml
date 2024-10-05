"""
test_fit_model.py
=================

This module contains comprehensive unit tests for the fit_model module,
including tests for the BaseModel, ClassifierModel, RegressionModel classes,
and the fit function. It covers various scenarios including different model types,
confound correction methods, and edge cases.

Test Summary:
1. Test Model Fitting
2. Test Feature / Target Handling
3. Test Performance Evaluation
4. Test Score Recording
5. Test Error Handling
6. Test Memory Efficiency
7. Test Different Data Types
8. Test Reproducibility
9. Test Edge Cases
10. Test Existing Scores Reuse
"""

import json
from typing import Dict, Any, List
import h5py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from workflow.scripts.fit_model import (
    ClassifierModel,
    RegressionModel,
    fit,
    BaseModel,
    MODELS
)

@pytest.fixture
def create_h5_dataset(tmpdir):
    """Fixture to create an HDF5 dataset for testing."""
    def _create_h5_dataset(data, filename):
        file_path = tmpdir.join(filename)
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('mask', data=np.ones(data.shape[0], dtype=bool))
        return str(file_path)
    return _create_h5_dataset

# 1. Test Model Fitting

@pytest.mark.parametrize("model_type", ["classification", "regression"])
def test_model_fitting(create_h5_dataset, model_type):
    """Test correct fitting of different model types with various hyperparameters."""
    if model_type == "classification":
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        model_class = ClassifierModel(lambda **args: LogisticRegression(**args), "logistic_regression")
        param_name = "C"
    else:
        X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        model_class = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")
        param_name = "alpha"

    X, y = X * 100, y * 50  # Scale features and targets
    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        for param_value in [0.1, 1.0, 10.0]:  # Test different hyperparameters
            metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal", **{param_name: param_value})
            assert isinstance(metrics, dict), f"Expected dict of metrics, got {type(metrics)}"
            assert all(0 <= v <= 1 for k, v in metrics.items() if k.startswith(('acc', 'f1', 'r2'))), "Metrics out of expected range"

# 2. Test Feature / Target Handling

@pytest.mark.parametrize("mode", ["normal", "with_cni", "only_cni"])
def test_feature_handling(create_h5_dataset, mode):
    """Test correct handling of features for different modes."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        model_class = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode=mode)

        assert isinstance(metrics, dict), f"Expected dict of metrics, got {type(metrics)}"
        assert all(k in metrics for k in ['r2_train', 'r2_val', 'r2_test']), "Missing expected metrics"

# 3. Test Performance Evaluation

def test_performance_evaluation(create_h5_dataset):
    """Test correct computation of metrics for classification and regression tasks."""
    # Classification
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    x_path = create_h5_dataset(X, 'features_cls.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets_cls.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni_cls.h5')

    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        cls_model = ClassifierModel(lambda **args: LogisticRegression(**args), "logistic_regression")
        cls_metrics = cls_model.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

        assert all(k in cls_metrics for k in ['acc_train', 'acc_val', 'acc_test', 'f1_train', 'f1_val', 'f1_test'])

    # Regression
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    x_path = create_h5_dataset(X, 'features_reg.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets_reg.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni_reg.h5')

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        reg_model = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")
        reg_metrics = reg_model.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

        assert all(k in reg_metrics for k in ['r2_train', 'r2_val', 'r2_test', 'mae_train', 'mae_val', 'mae_test', 'mse_train', 'mse_val', 'mse_test'])

# 4. Test Score Recording

def test_score_recording(tmpdir, create_h5_dataset):
    """Test correct recording of scores in CSV format."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    split_path = tmpdir.join("split.json")
    with open(split_path, "w") as f:
        json.dump({
            "idx_train": list(range(0, 60)),
            "idx_val": list(range(60, 80)),
            "idx_test": list(range(80, 100)),
            "samplesize": 100,
            "seed": 42,
        }, f)

    scores_path = tmpdir.join("scores.csv")
    
    fit(
        x_path,
        y_path,
        str(split_path),
        str(scores_path),
        "ridge-reg",
        {"ridge-reg": {"alpha": [0.1, 1.0, 10.0]}},
        [],
        "normal",
        cni_path,
    )

    scores = pd.read_csv(scores_path)
    assert set(scores.columns) == {'alpha', 'r2_train', 'r2_val', 'r2_test', 'mae_train', 'mae_val', 'mae_test', 'mse_train', 'mse_val', 'mse_test', 'n', 's'}
    assert len(scores) == 3  # One row for each alpha value

# 5. Test Error Handling

def test_error_handling(create_h5_dataset):
    """Test appropriate error messages for invalid inputs or configurations."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        model_class = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")

        with pytest.raises(ValueError):
            model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="invalid_mode")

# 6. Test Memory Efficiency

def test_memory_efficiency(create_h5_dataset):
    """Test efficient data loading and processing for large datasets."""
    X, y = make_regression(n_samples=10000, n_features=100, random_state=42)
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(10000, 3), 'cni.h5')

    idx_train, idx_val, idx_test = list(range(0, 6000)), list(range(6000, 8000)), list(range(8000, 10000))

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        model_class = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

        assert isinstance(metrics, dict), "Failed to process large dataset efficiently"

# 7. Test Different Data Types

@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_different_data_types(create_h5_dataset, dtype):
    """Test correct handling of various data types."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X, y = X.astype(dtype), y.astype(dtype)
    
    x_path = create_h5_dataset(X, f'features_{dtype.__name__}.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), f'targets_{dtype.__name__}.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3).astype(dtype), f'cni_{dtype.__name__}.h5')

    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        model_class = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

        assert isinstance(metrics, dict), f"Failed to handle {dtype.__name__} data type"

# 8. Test Reproducibility

def test_reproducibility(create_h5_dataset):
    """Test consistent results with the same random seed."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        model_class = RegressionModel(lambda **args: Ridge(random_state=42, **args), "ridge_regression")
        metrics1 = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")
        metrics2 = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

        assert metrics1 == metrics2, "Results are not reproducible with the same random seed"

# 9. Test Edge Cases

def test_edge_cases(create_h5_dataset):
    """Test behavior with minimal datasets and imbalanced datasets."""
    # Minimal dataset
    X, y = make_classification(n_samples=10, n_features=5, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=42)
    x_path = create_h5_dataset(X, 'features_minimal.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets_minimal.h5')
    cni_path = create_h5_dataset(np.random.rand(10, 2), 'cni_minimal.h5')

    idx_train, idx_val, idx_test = [0, 1, 2, 3, 4], [5, 6, 7], [8, 9]

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        model_class = ClassifierModel(lambda **args: LogisticRegression(**args), "logistic_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

        assert isinstance(metrics, dict), "Failed to handle minimal dataset"

    # Imbalanced dataset
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)
    x_path = create_h5_dataset(X, 'features_imbalanced.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets_imbalanced.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni_imbalanced.h5')

    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset, y_dataset, cni_dataset = x_file['data'], y_file['data'], cni_file['data']

        model_class = ClassifierModel(lambda **args: LogisticRegression(**args), "logistic_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

        assert isinstance(metrics, dict), "Failed to handle imbalanced dataset"

# 10. Test Existing Scores Reuse

def test_existing_scores_reuse(tmpdir, create_h5_dataset):
    """Test that existing scores are successfully reused."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    split_path = tmpdir.join("split.json")
    with open(split_path, "w") as f:
        json.dump({
            "idx_train": list(range(0, 60)),
            "idx_val": list(range(60, 80)),
            "idx_test": list(range(80, 100)),
            "samplesize": 100,
            "seed": 42,
        }, f)

    # Create existing scores
    existing_scores_path = tmpdir.join("existing_scores.csv")
    existing_scores = pd.DataFrame({
        'alpha': [0.1, 1.0],
        'r2_train': [0.8, 0.7],
        'r2_val': [0.75, 0.65],
        'r2_test': [0.7, 0.6],
        'mae_train': [0.2, 0.25],
        'mae_val': [0.22, 0.27],
        'mae_test': [0.24, 0.29],
        'mse_train': [0.04, 0.0625],
        'mse_val': [0.0484, 0.0729],
        'mse_test': [0.0576, 0.0841],
        'n': [100, 100],
        's': [42, 42]
    })
    existing_scores.to_csv(existing_scores_path, index=False)

    scores_path = tmpdir.join("scores.csv")
    
    # Run fit function with existing scores
    new_scores = fit(
        x_path,
        y_path,
        str(split_path),
        str(scores_path),
        "ridge-reg",
        {"ridge-reg": {"alpha": [0.1, 1.0, 10.0]}},
        [str(existing_scores_path)],
        "normal",
        cni_path,
    )

    # Check that existing scores were reused
    assert len(new_scores) == 3, "Expected 3 rows in new_scores (2 existing + 1 new)"
    
    # Compare existing scores with a tolerance, ignoring data types
    pd.testing.assert_frame_equal(
        new_scores.iloc[:2].astype(float), 
        existing_scores.astype(float), 
        check_exact=False, 
        rtol=1e-5, 
        atol=1e-8
    )
    
    assert np.isclose(new_scores.iloc[2]['alpha'], 10.0), "Third row should have alpha close to 10.0 (new computation)"

    # Verify that the CSV file was written correctly
    saved_scores = pd.read_csv(scores_path)
    pd.testing.assert_frame_equal(
        saved_scores.astype(float), 
        new_scores.astype(float), 
        check_exact=False, 
        rtol=1e-5, 
        atol=1e-8
    )

# Add this new test function
@pytest.mark.parametrize("confound_correction_method", ["with_cni", "only_cni", "normal"])
@pytest.mark.parametrize("model_type", ["classification", "regression"])
def test_fit_with_confound_correction(tmpdir, create_h5_dataset, confound_correction_method, model_type):
    """
    Test the fit function with various confound correction methods and model types.

    This test creates toy datasets, sets up the necessary file structure,
    runs the fit function, and checks the output for correctness.

    Args:
        tmpdir (str): Pytest fixture for temporary directory.
        create_h5_dataset (callable): Fixture to create HDF5 datasets.
        confound_correction_method (str): Method for confound correction.
        model_type (str): Type of model to test (classification or regression).
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
    split = {
        "idx_train": list(range(0, 60)),
        "idx_val": list(range(60, 80)),
        "idx_test": list(range(80, 100)),
        "samplesize": 100,
        "seed": 42,
    }

    # Save data to temporary files
    features_path = create_h5_dataset(X, 'features.h5')
    targets_path = create_h5_dataset(y, 'targets.h5')
    cni_path = create_h5_dataset(cni, 'cni.h5')
    split_path = tmpdir.join("split.json")
    with open(split_path, "w") as f:
        json.dump(split, f)

    # Define model parameters
    hp = {"alpha": [0.1, 1.0, 10.0]}
    grid = {model_name: hp}

    # Run the fit function
    scores_path = tmpdir.join("scores.csv")
    fit(
        features_path,
        targets_path,
        str(split_path),
        str(scores_path),
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
    expected_columns = ["n", "s", "alpha"]
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