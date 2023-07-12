import os

import numpy as np
import pytest

from esce.confound_regression import confound_regression


@pytest.mark.parametrize(
    ("data", "confounds", "expected"),
    [
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.zeros((5, 1)),
        ),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]), np.zeros((2, 2))),
    ],
)
def test_confound_regression(data, confounds, expected):
    out_path = "corrected.npy"

    # Save test data
    np.save("data.npy", data)
    np.save("confounds.npy", confounds)

    # Run the function
    confound_regression("data.npy", "confounds.npy", out_path)

    # check if the output file was created
    assert os.path.exists(out_path)

    # Load the corrected data
    corrected_data = np.load(out_path)

    # Check output shape
    assert corrected_data.shape == expected.shape

    # Check if the corrected data is as expected
    assert np.allclose(corrected_data, expected)

    # Clean up
    os.remove("data.npy")
    os.remove("confounds.npy")
    os.remove(out_path)
