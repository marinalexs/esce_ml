"""
test_predefined_datasets.py
===========================

This module contains tests for the predefined datasets functionality in the ESCE workflow.
It focuses on testing the MNIST dataset preparation, which is one of the predefined datasets
available in the workflow.

Test Summary:
1. test_mnist_features: Tests the preparation of MNIST features.
2. test_mnist_targets: Tests the preparation of MNIST targets.

These tests are marked as slow and are skipped by default. To run these tests,
remove the @pytest.mark.skip decorator.

Note: These tests require downloading the MNIST dataset, which may take some time
depending on your internet connection.
"""

import h5py
import pytest

from workflow.scripts.prepare_data import prepare_data

@pytest.mark.slow()
@pytest.mark.skip(reason="Disabled by default. Remove this line to run the test.")
def test_mnist_features(tmpdir):
    """
    Test the preparation of MNIST features.

    This test checks if the MNIST features are correctly prepared and stored
    in the expected format.

    Args:
        tmpdir: Pytest fixture for temporary directory.

    Raises:
        AssertionError: If the output data shape doesn't match the expected shape.
    """
    out_path = str(tmpdir / "test.h5")
    dataset = "mnist"
    features_targets_covariates = "features"
    variant = "pixel"
    custom_datasets = {}

    # Prepare MNIST features data
    prepare_data(
        out_path, dataset, features_targets_covariates, variant, custom_datasets
    )

    # Load the data from the output file and check dimensions
    with h5py.File(out_path, "r") as f:
        output_data = f["data"][:]

    # MNIST has 70,000 samples (60,000 training + 10,000 test) and 784 features (28x28 pixels)
    assert output_data.shape == (70000, 784), "Unexpected shape for MNIST features"

@pytest.mark.slow()
@pytest.mark.skip(reason="Disabled by default. Remove this line to run the test.")
def test_mnist_targets(tmpdir):
    """
    Test the preparation of MNIST targets.

    This test checks if the MNIST targets are correctly prepared and stored
    in the expected format.

    Args:
        tmpdir: Pytest fixture for temporary directory.

    Raises:
        AssertionError: If the output data shape, data type, or target values don't match the expected values.
    """
    out_path = str(tmpdir / "test.h5")
    dataset = "mnist"
    features_targets_covariates = "targets"
    variant = "ten-digits"
    custom_datasets = {}

    # Prepare MNIST targets data
    prepare_data(
        out_path, dataset, features_targets_covariates, variant, custom_datasets
    )

    # Load the data from the output file and check dimensions
    with h5py.File(out_path, "r") as f:
        output_data = f["data"][:]

    # MNIST has 70,000 samples (60,000 training + 10,000 test) and 1 target (digit label)
    assert output_data.shape == (70000,), "Unexpected shape for MNIST targets"
    assert output_data.dtype == int, "Unexpected data type for MNIST targets"
    assert set(output_data) == set(range(10)), "Unexpected target values for MNIST"
