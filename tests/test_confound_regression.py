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

import numpy as np
import pytest
from typing import Tuple, List, Callable, Dict
from pathlib import Path
from numpy.typing import NDArray

from workflow.scripts.confound_regression import confound_regression, load_h5_data


@pytest.mark.parametrize(
    ("data", "confounds", "expected", "expected_mask"),
    [
        # Test case 1: Simple 2D data and confounds
        (
            np.array([[1], [2], [3], [4], [5]]),
            np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]),
            np.array([[0], [0], [0], [0], [0]]),
            np.array([True, True, True, True, True]),
        ),
        # Test case 2: 2D data and confounds with multiple features
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([[0, 0], [0, 0], [0, 0]]),
            np.array([True, True, True]),
        ),
        # Test case 3: Data with missing values (masked)
        (
            np.array([[1], [2], [np.nan], [4], [5]]),
            np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]),
            np.array([[0], [0], [np.nan], [0], [0]]),
            np.array([True, True, False, True, True]),
        ),
        # Test case 4: Confounds with missing values (masked)
        (
            np.array([[1], [2], [3], [4], [5]]),
            np.array([[0.1], [0.2], [0.3], [0.4], [np.nan]]),
            np.array([[0], [0], [0], [0], [np.nan]]),
            np.array([True, True, True, True, False]),
        ),
    ],
)
def test_confound_regression(
    tmp_path: Path,
    write_data_to_file: Callable,
    data: NDArray,
    confounds: NDArray,
    expected: NDArray,
    expected_mask: NDArray,
) -> None:
    """
    Test the confound_regression function with various input scenarios.

    This test creates input files with different data shapes and masks,
    runs the confound regression, and checks if the output is as expected.

    Args:
        tmp_path (Path): Temporary directory for creating test files.
        write_data_to_file (Callable): Fixture to write data to a file.
        data (NDArray): Input data array (2D).
        confounds (NDArray): Input confounds array (2D).
        expected (NDArray): Expected output after confound regression (1D).
        data_mask (NDArray): Mask for input data (1D).
        confounds_mask (NDArray): Mask for confounds data (1D).
        expected_mask (NDArray): Expected mask for corrected data (1D).
    """
    # Create input files
    data_path = write_data_to_file(data, 'h5', tmp_path / "data.h5")
    confound_path = write_data_to_file(confounds, 'h5', tmp_path / "confounds.h5")
    out_path = tmp_path / "corrected.h5"

    # Run the confound regression
    confound_regression(str(data_path), str(confound_path), str(out_path))

    # Load the corrected data
    corrected_data, corrected_mask = load_h5_data(str(out_path))

    # Check output shape
    assert corrected_data.shape == expected.shape, "Output shape does not match expected shape"

    # Check if the corrected data is as expected
    np.testing.assert_allclose(corrected_data, expected, rtol=1e-5, atol=1e-8, 
                               err_msg="Corrected data does not match expected values")

    # Check if the mask is correct
    np.testing.assert_array_equal(corrected_mask, expected_mask, 
                                  err_msg="Corrected mask does not match expected mask")


def test_confound_regression_error_handling(
    tmp_path: Path,
    write_data_to_file: Callable
) -> None:
    """
    Test error handling in confound_regression function.

    This test checks if an AssertionError is raised when the data and
    confounds have mismatched shapes.

    Args:
        tmp_path (Path): Temporary directory for creating test files.
        write_data_to_file (Callable): Fixture to write data to a file.
    """
    # Create mismatched data and confounds
    data = np.array([[1], [2], [3], [4], [5]])
    confounds = np.array([[0.1], [0.2], [0.3], [0.4]])  # One less element than data

    data_path = write_data_to_file(data, 'h5', tmp_path / "data.h5")
    confound_path = write_data_to_file(confounds, 'h5', tmp_path / "confounds.h5")
    out_path = tmp_path / "corrected.h5"

    # Check if AssertionError is raised for mismatched data and confounds
    with pytest.raises(AssertionError):
        confound_regression(str(data_path), str(confound_path), str(out_path))