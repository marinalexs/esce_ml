"""
test_confound_regression.py
===========================

This module contains unit tests for the confound_regression function from the
confound_regression module. It tests various scenarios including different
data shapes, missing values, and error handling.

Test Summary:
1. test_confound_regression: Tests the confound_regression function with various input scenarios.
2. test_confound_regression_error_handling: Tests error handling for mismatched data and confounds.

The tests use parametrization to cover multiple scenarios and check if the
confound regression is performed correctly under different conditions.
"""

import h5py
import numpy as np
import pytest
from typing import Tuple, List
from numpy.typing import NDArray

from workflow.scripts.confound_regression import confound_regression


def create_h5_file(tmpdir: str, filename: str, data: NDArray, mask: NDArray) -> str:
    """
    Helper function to create an H5 file with data and mask.

    Args:
        tmpdir (str): Temporary directory path.
        filename (str): Name of the file to be created.
        data (NDArray): Data to be stored in the file.
        mask (NDArray): Mask to be stored in the file.

    Returns:
        str: Path to the created H5 file.
    """
    file_path = str(tmpdir / filename)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("mask", data=mask)
    return file_path


@pytest.mark.parametrize(
    ("data", "confounds", "expected", "data_mask", "confounds_mask"),
    [
        # Test case 1: Simple 1D data and confounds
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([[0], [0], [0], [0], [0]]),
            np.array([True, True, True, True, True]),
            np.array([True, True, True, True, True]),
        ),
        # Test case 2: 2D data and confounds
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([True, True, True]),
            np.array([True, True, True]),
        ),
        # Test case 3: Data with missing values (masked)
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([[0], [0], [np.nan], [0], [0]]),
            np.array([True, True, False, True, True]),
            np.array([True, True, True, True, True]),
        ),
        # Test case 4: Confounds with missing values (masked)
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([[0], [0], [np.nan], [0], [0]]),
            np.array([True, True, True, True, True]),
            np.array([True, True, False, True, True]),
        ),
    ],
)
def test_confound_regression(
    tmpdir: str,
    data: NDArray,
    confounds: NDArray,
    expected: NDArray,
    data_mask: NDArray,
    confounds_mask: NDArray,
) -> None:
    """
    Test the confound_regression function with various input scenarios.

    This test creates input files with different data shapes and masks,
    runs the confound regression, and checks if the output is as expected.

    Args:
        tmpdir (str): Temporary directory for creating test files.
        data (NDArray): Input data array.
        confounds (NDArray): Input confounds array.
        expected (NDArray): Expected output after confound regression.
        data_mask (NDArray): Mask for input data.
        confounds_mask (NDArray): Mask for confounds data.
    """
    # Create input files
    data_path = create_h5_file(tmpdir, "data.h5", data, data_mask)
    confound_path = create_h5_file(tmpdir, "confounds.h5", confounds, confounds_mask)
    out_path = str(tmpdir / "corrected.h5")

    # Run the confound regression
    confound_regression(data_path, confound_path, out_path)

    # Load the corrected data
    with h5py.File(out_path, "r") as f:
        corrected_data = f["data"][:]
        corrected_mask = f["mask"][:]

    # Check output shape
    assert corrected_data.shape == expected.shape, "Output shape does not match expected shape"

    # Check if the corrected data is as expected
    np.testing.assert_allclose(corrected_data, expected, rtol=1e-5, atol=1e-8, 
                               err_msg="Corrected data does not match expected values")

    # Check if the mask is correct
    expected_mask = np.logical_and(data_mask, confounds_mask)
    np.testing.assert_array_equal(corrected_mask, expected_mask, 
                                  err_msg="Corrected mask does not match expected mask")


def test_confound_regression_error_handling(tmpdir: str) -> None:
    """
    Test error handling in confound_regression function.

    This test checks if an AssertionError is raised when the data and
    confounds have mismatched shapes.

    Args:
        tmpdir (str): Temporary directory for creating test files.
    """
    # Create mismatched data and confounds
    data = np.array([1, 2, 3, 4, 5])
    confounds = np.array([0.1, 0.2, 0.3, 0.4])  # One less element than data
    mask = np.array([True, True, True, True, True])

    data_path = create_h5_file(tmpdir, "data.h5", data, mask)
    confound_path = create_h5_file(tmpdir, "confounds.h5", confounds, mask[:-1])
    out_path = str(tmpdir / "corrected.h5")

    # Check if AssertionError is raised for mismatched data and confounds
    with pytest.raises(AssertionError):
        confound_regression(data_path, confound_path, out_path)
