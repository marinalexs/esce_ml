"""
test_prepare_data.py
====================

This module contains unit tests for the prepare_data function in the prepare_data module
of the ESCE workflow. It tests the functionality of preparing custom datasets with
various input file types and data categories.

Test Summary:
1. test_prepare_data_custom_datasets: Parameterized test for different input file types
   and data categories (features, targets, covariates).

These tests ensure that the prepare_data function correctly handles different input
formats and properly converts them to the required HDF5 format.
"""

import h5py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from workflow.scripts.prepare_data import prepare_data, predefined_datasets

TEST_CASES = [
    ("tsv", "targets"),
    ("tsv", "features"),
    ("tsv", "covariates"),
    ("csv", "targets"),
    ("csv", "features"),
    ("csv", "covariates"),
    ("npy", "targets"),
    ("npy", "features"),
    ("npy", "covariates"),
]


@pytest.mark.parametrize(
    (
        "in_file_type",
        "features_targets_covariates",
    ),
    TEST_CASES,
)
def test_prepare_data_custom_datasets(
    tmpdir,
    in_file_type,
    features_targets_covariates,
):
    """
    Test the prepare_data function with custom datasets.

    This test creates dummy data in various formats (CSV, TSV, NPY) and for different
    data categories (features, targets, covariates). It then uses the prepare_data
    function to convert this data to the HDF5 format and checks if the output is correct.

    Args:
        tmpdir: Pytest fixture for temporary directory.
        in_file_type (str): Input file type (csv, tsv, or npy).
        features_targets_covariates (str): Data category (features, targets, or covariates).

    Raises:
        AssertionError: If the output data dimensions don't match the expected dimensions.
    """
    # Create input and output file paths
    in_path = str(tmpdir.join(f"data.{in_file_type}"))
    out_path = str(tmpdir.join("data.h5"))

    # Create dummy data
    dummy_data = (
        np.random.rand(10, 1)
        if features_targets_covariates == "targets"
        else np.random.rand(10, 2)
    )
    dummy_data = pd.DataFrame(dummy_data)

    # Save dummy data in the specified format
    if in_file_type == "csv":
        dummy_data.to_csv(in_path, index=False)
    elif in_file_type == "tsv":
        dummy_data.to_csv(in_path, sep="\t", index=False)
    elif in_file_type == "npy":
        np.save(in_path, dummy_data.values)

    # Create custom datasets dictionary
    custom_datasets = {
        "pytest": {
            "targets": {"normal": in_path},
            "features": {"normal": in_path},
            "covariates": {"normal": in_path},
        }
    }

    # Run prepare_data function
    prepare_data(
        out_path, "pytest", features_targets_covariates, "normal", custom_datasets
    )

    # Load the data from the output file and check its dimensions
    with h5py.File(out_path, "r") as f:
        output_data = f["data"][:]
        output_mask = f["mask"][:]

    expected_shape = 1 if features_targets_covariates == "targets" else 2

    # Check if the output data has the expected number of dimensions
    assert output_data.ndim == expected_shape, f"Expected {expected_shape} dimensions, but got {output_data.ndim}"
    
    # Check if the output mask is 1-dimensional
    assert output_mask.ndim == 1, f"Expected 1-dimensional mask, but got {output_mask.ndim}-dimensional"

    # Additional checks could be added here, e.g., checking the actual values of the data and mask
