"""
test_base_models.py
===================

This module contains unit tests for the ClassifierModel and RegressionModel classes
from the fit_model module. It tests various scenarios including different model types
(classification and regression) and confound correction methods.

Test Summary:
1. test_classifier_model: Tests the ClassifierModel with a binary classification problem.
2. test_regression_model: Tests the RegressionModel with a regression problem.
3. test_classifier_model_with_cni: Tests the ClassifierModel with confounds of no interest (CNI).
4. test_regression_model_only_cni: Tests the RegressionModel using only CNI as features.

Each test creates synthetic data, processes it through the respective model,
and checks if the output metrics are as expected.
"""

import pytest
import numpy as np
import h5py
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from workflow.scripts.fit_model import ClassifierModel, RegressionModel

@pytest.fixture
def create_h5_dataset(tmpdir):
    """
    Fixture to create an HDF5 dataset for testing.

    Args:
        tmpdir: Pytest fixture for temporary directory.

    Returns:
        function: A function to create HDF5 datasets.
    """
    def _create_h5_dataset(data, filename):
        file_path = tmpdir.join(filename)
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('mask', data=np.ones(data.shape[0], dtype=bool))
        return str(file_path)
    return _create_h5_dataset

def test_classifier_model(create_h5_dataset):
    """
    Test the ClassifierModel with a binary classification problem.

    This test creates synthetic classification data, processes it through
    the ClassifierModel, and checks if the output metrics are as expected.

    Args:
        create_h5_dataset (function): Fixture to create HDF5 datasets.
    """
    # Create a toy binary classification problem
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    X = X * 100  # Scale features to make the problem more challenging

    # Define indices for train, validation, and test sets
    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    # Create HDF5 datasets for features, targets, and confounds
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    # Open HDF5 files and process data through the ClassifierModel
    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset = x_file['data']
        y_dataset = y_file['data']
        cni_dataset = cni_file['data']

        model_class = ClassifierModel(lambda **args: LogisticRegression(**args), "logistic_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

    # Check if all expected metrics are present and within expected ranges
    expected_metrics = ["acc_train", "acc_val", "acc_test", "f1_train", "f1_val", "f1_test"]
    for metric in expected_metrics:
        assert metric in metrics, f"Expected metric {metric} not found in results"
        assert 0 <= metrics[metric] <= 1, f"Metric {metric} is out of expected range [0, 1]"

def test_regression_model(create_h5_dataset):
    """
    Test the RegressionModel with a regression problem.

    This test creates synthetic regression data, processes it through
    the RegressionModel, and checks if the output metrics are as expected.

    Args:
        create_h5_dataset (function): Fixture to create HDF5 datasets.
    """
    # Create a toy regression problem
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X, y = X * 100, y * 50  # Scale features and targets to make the problem more challenging

    # Define indices for train, validation, and test sets
    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    # Create HDF5 datasets for features, targets, and confounds
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    # Open HDF5 files and process data through the RegressionModel
    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset = x_file['data']
        y_dataset = y_file['data']
        cni_dataset = cni_file['data']

        model_class = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="normal")

    # Check if all expected metrics are present and within expected ranges
    expected_metrics = ["r2_train", "r2_val", "r2_test", "mae_train", "mae_val", "mae_test", "mse_train", "mse_val", "mse_test"]
    for metric in expected_metrics:
        assert metric in metrics, f"Expected metric {metric} not found in results"
        if metric.startswith("r2"):
            assert -1 <= metrics[metric] <= 1, f"Metric {metric} is out of expected range [-1, 1]"
        else:
            assert metrics[metric] >= 0, f"Metric {metric} should be non-negative"

def test_classifier_model_with_cni(create_h5_dataset):
    """
    Test the ClassifierModel with confounds of no interest (CNI).

    This test creates synthetic classification data, processes it through
    the ClassifierModel using the 'with_cni' mode, and checks if the output
    metrics are as expected.

    Args:
        create_h5_dataset (function): Fixture to create HDF5 datasets.
    """
    # Create a toy binary classification problem
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    X = X * 100  # Scale features to make the problem more challenging

    # Define indices for train, validation, and test sets
    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    # Create HDF5 datasets for features, targets, and confounds
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    # Open HDF5 files and process data through the ClassifierModel with CNI
    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset = x_file['data']
        y_dataset = y_file['data']
        cni_dataset = cni_file['data']

        model_class = ClassifierModel(lambda **args: LogisticRegression(**args), "logistic_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="with_cni")

    # Check if all expected metrics are present and within expected ranges
    expected_metrics = ["acc_train", "acc_val", "acc_test", "f1_train", "f1_val", "f1_test"]
    for metric in expected_metrics:
        assert metric in metrics, f"Expected metric {metric} not found in results"
        assert 0 <= metrics[metric] <= 1, f"Metric {metric} is out of expected range [0, 1]"

def test_regression_model_only_cni(create_h5_dataset):
    """
    Test the RegressionModel using only confounds of no interest (CNI) as features.

    This test creates synthetic regression data, processes it through
    the RegressionModel using the 'only_cni' mode, and checks if the output
    metrics are as expected.

    Args:
        create_h5_dataset (function): Fixture to create HDF5 datasets.
    """
    # Create a toy regression problem
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X, y = X * 100, y * 50  # Scale features and targets to make the problem more challenging

    # Define indices for train, validation, and test sets
    idx_train, idx_val, idx_test = list(range(0, 60)), list(range(60, 80)), list(range(80, 100))

    # Create HDF5 datasets for features, targets, and confounds
    x_path = create_h5_dataset(X, 'features.h5')
    y_path = create_h5_dataset(y.reshape(-1, 1), 'targets.h5')
    cni_path = create_h5_dataset(np.random.rand(100, 3), 'cni.h5')

    # Open HDF5 files and process data through the RegressionModel using only CNI
    with h5py.File(x_path, 'r') as x_file, h5py.File(y_path, 'r') as y_file, h5py.File(cni_path, 'r') as cni_file:
        x_dataset = x_file['data']
        y_dataset = y_file['data']
        cni_dataset = cni_file['data']

        model_class = RegressionModel(lambda **args: Ridge(**args), "ridge_regression")
        metrics = model_class.score(x_dataset, y_dataset, cni_dataset, idx_train, idx_val, idx_test, mode="only_cni")

    # Check if all expected metrics are present and within expected ranges
    expected_metrics = ["r2_train", "r2_val", "r2_test", "mae_train", "mae_val", "mae_test", "mse_train", "mse_val", "mse_test"]
    for metric in expected_metrics:
        assert metric in metrics, f"Expected metric {metric} not found in results"
        if metric.startswith("r2"):
            assert -1 <= metrics[metric] <= 1, f"Metric {metric} is out of expected range [-1, 1]"
        else:
            assert metrics[metric] >= 0, f"Metric {metric} should be non-negative"