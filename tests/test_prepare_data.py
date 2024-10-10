"""
test_prepare_data.py
====================

This module contains unit tests for the prepare_data function in the prepare_data module
of the ESCE workflow. It tests the functionality of preparing both custom and predefined datasets
with various input file types and data categories.

The tests use fixtures from conftest.py to generate synthetic data and create datasets.

Test Summary:
1. test_prepare_data_custom_datasets: Tests custom dataset preparation for different file types and data categories.
2. test_prepare_data_empty_covariates: Tests handling of empty covariate datasets.
3. test_prepare_data_errors: Tests error handling for various edge cases.
4. test_prepare_data_predefined_dataset: Tests preparation of predefined datasets (MNIST features).
5. test_prepare_data_missing_dataset: Tests error handling for non-existent datasets.
6. test_mnist_features: Tests preparation of MNIST features.
7. test_mnist_targets: Tests preparation of MNIST targets for different variants.
8. test_mnist_covariates: Tests preparation of MNIST covariates (which should be empty).
"""

import h5py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from typing import Dict, Callable

from workflow.scripts.prepare_data import prepare_data, predefined_datasets, FLOAT_PRECISION

@pytest.fixture
def custom_datasets(tmpdir: Path, generate_synth_data: Callable, create_dataset: Callable) -> Dict[str, Dict]:
    """
    Fixture to create custom datasets for testing.

    Args:
        tmpdir: Temporary directory for test files.
        generate_synth_data: Fixture to generate synthetic data.
        create_dataset: Fixture to create dataset files.

    Returns:
        Dict containing custom datasets in different formats (csv, tsv, npy, parquet, h5).
    """
    X, y, confounds = generate_synth_data(n_samples=100, n_features=10)
    formats = ['csv', 'tsv', 'npy','parquet', 'h5']
    datasets = {}

    for fmt in formats:
        base_path = Path(tmpdir) / f"data_{fmt}"
        datasets[fmt] = create_dataset(X, y, confounds, fmt, base_path)
        datasets[fmt] = {k: {'normal': str(v)} for k, v in datasets[fmt].items()}

    return datasets

def test_prepare_data_custom_datasets(tmpdir: Path, custom_datasets: Dict[str, Dict]):
    """
    Test preparation of custom datasets for different input file types and data categories.
    
    This test checks if the prepare_data function correctly processes custom datasets
    in various formats (csv, tsv, parquet, h5) for features, targets, and covariates.
    """
    for fmt in custom_datasets:
        for data_type in ['features', 'targets', 'covariates']:
            out_path = str(tmpdir / f"{data_type}_{fmt}.h5")
            prepare_data(
                out_path=out_path,
                dataset='custom',
                features_targets_covariates=data_type,
                variant='normal',
                custom_datasets={'custom': {
                    'features': custom_datasets[fmt]['features'],
                    'targets': custom_datasets[fmt]['targets'],
                    'covariates': custom_datasets[fmt]['confounds']  # Note the change here
                }}
            )

            with h5py.File(out_path, 'r') as f:
                assert 'data' in f and 'mask' in f, f"Missing data or mask for {data_type} in {fmt} format"
                assert f['data'].dtype == FLOAT_PRECISION, f"Incorrect data type for {data_type} in {fmt} format"
                assert f['mask'].dtype == bool, f"Incorrect mask type for {data_type} in {fmt} format"

                if data_type == 'targets':
                    assert f['data'].ndim == 1, f"Incorrect dimension for targets in {fmt} format"
                elif data_type == 'features':
                    assert f['data'].shape[1] == 10, f"Incorrect number of features in {fmt} format"
                elif data_type == 'covariates':
                    assert f['data'].shape[1] == 3, f"Incorrect number of covariates in {fmt} format"

def test_prepare_data_empty_covariates(tmpdir: Path):
    """
    Test handling of empty covariate datasets.
    
    This test ensures that the prepare_data function correctly handles
    the case when no covariates are provided.
    """
    out_path = str(tmpdir / "empty_covariates.h5")
    prepare_data(
        out_path=out_path,
        dataset='custom',
        features_targets_covariates='covariates',
        variant='none',
        custom_datasets={}
    )

    with h5py.File(out_path, 'r') as f:
        assert 'data' in f and 'mask' in f, "Missing data or mask for empty covariates"
        assert f['data'].shape == (0,), "Incorrect shape for empty covariates data"
        assert f['mask'].shape == (0,), "Incorrect shape for empty covariates mask"

def test_prepare_data_errors(tmpdir: Path):
    """
    Test error handling for various edge cases.
    
    This test checks if the prepare_data function correctly raises
    exceptions for unsupported file formats, empty datasets, and
    incompatible data types.
    """
    # Test unsupported file format
    with pytest.raises(ValueError, match="Unsupported file format"):
        prepare_data(
            out_path=str(tmpdir / "error.h5"),
            dataset='custom',
            features_targets_covariates='features',
            variant='normal',
            custom_datasets={'custom': {'features': {'normal': 'data.txt'}}}
        )

    # Test empty dataset
    empty_file = str(tmpdir / "empty.csv")
    pd.DataFrame().to_csv(empty_file, index=False)
    with pytest.raises((ValueError, pd.errors.EmptyDataError), match="No columns to parse from file|Dataset is empty after loading"):
        prepare_data(
            out_path=str(tmpdir / "error.h5"),
            dataset='custom',
            features_targets_covariates='features',
            variant='normal',
            custom_datasets={'custom': {'features': {'normal': empty_file}}}
        )

    # Test incompatible data types
    string_data = str(tmpdir / "string_data.csv")
    pd.DataFrame({'col': ['a', 'b', 'c']}).to_csv(string_data, index=False)
    with pytest.raises(TypeError, match="Incompatible data type"):
        prepare_data(
            out_path=str(tmpdir / "error.h5"),
            dataset='custom',
            features_targets_covariates='features',
            variant='normal',
            custom_datasets={'custom': {'features': {'normal': string_data}}}
        )

def test_prepare_data_predefined_dataset(tmpdir: Path):
    """
    Test preparation of predefined datasets (MNIST features).
    
    This test checks if the prepare_data function correctly processes
    the predefined MNIST dataset for features.
    """
    out_path = str(tmpdir / "mnist_features.h5")
    prepare_data(
        out_path=out_path,
        dataset='mnist',
        features_targets_covariates='features',
        variant='pixel',
        custom_datasets={}
    )

    with h5py.File(out_path, 'r') as f:
        assert 'data' in f and 'mask' in f, "Missing data or mask for MNIST features"
        assert f['data'].shape[1] == 784, "Incorrect number of features for MNIST dataset"

def test_prepare_data_missing_dataset(tmpdir: Path):
    """
    Test error handling for non-existent datasets.
    
    This test ensures that the prepare_data function raises an appropriate
    exception when a non-existent dataset is requested.
    """
    with pytest.raises(KeyError, match="Requested predefined dataset or variant does not exist"):
        prepare_data(
            out_path=str(tmpdir / "error.h5"),
            dataset='nonexistent',
            features_targets_covariates='features',
            variant='normal',
            custom_datasets={}
        )

@pytest.mark.slow
def test_mnist_features(tmpdir: Path):
    """
    Test preparation of MNIST features.
    
    This test checks if the prepare_data function correctly processes
    the MNIST dataset features, including data shape, type, and value range.
    """
    out_path = str(tmpdir / "mnist_features.h5")
    prepare_data(
        out_path=out_path,
        dataset="mnist",
        features_targets_covariates="features",
        variant="pixel",
        custom_datasets={}
    )

    with h5py.File(out_path, "r") as f:
        assert 'data' in f and 'mask' in f, "Missing data or mask for MNIST features"
        assert f['data'].dtype == FLOAT_PRECISION, "Incorrect data type for MNIST features"
        assert f['mask'].dtype == bool, "Incorrect mask type for MNIST features"
        assert f['data'].shape == (70000, 784), "Incorrect shape for MNIST features"
        assert f['mask'].shape == (70000,), "Incorrect shape for MNIST features mask"
        data_array = f['data'][:]
        assert np.all((0 <= data_array) & (data_array <= 255)), "MNIST pixel values out of expected range"

@pytest.mark.slow
@pytest.mark.parametrize("variant, expected_classes", [
    ('ten-digits', set(range(10))),
    ('odd-even', {0, 1})
])
def test_mnist_targets(tmpdir: Path, variant: str, expected_classes: set):
    """
    Test preparation of MNIST targets for different variants.
    
    This test checks if the prepare_data function correctly processes
    MNIST targets for both 'ten-digits' and 'odd-even' variants.
    
    Args:
        variant: The MNIST target variant to test ('ten-digits' or 'odd-even').
        expected_classes: The expected set of unique target values.
    """
    out_path = str(tmpdir / f"mnist_targets_{variant}.h5")
    prepare_data(
        out_path=out_path,
        dataset="mnist",
        features_targets_covariates="targets",
        variant=variant,
        custom_datasets={}
    )

    with h5py.File(out_path, "r") as f:
        assert 'data' in f and 'mask' in f, f"Missing data or mask for MNIST {variant} targets"
        assert f['data'].dtype == FLOAT_PRECISION, f"Incorrect data type for MNIST {variant} targets"
        assert f['mask'].dtype == bool, f"Incorrect mask type for MNIST {variant} targets"
        assert f['data'].shape == (70000,), f"Incorrect shape for MNIST {variant} targets"
        assert f['mask'].shape == (70000,), f"Incorrect shape for MNIST {variant} targets mask"
        assert set(np.unique(f['data'])) == expected_classes, f"Unexpected target classes for MNIST {variant} variant"

def test_mnist_covariates(tmpdir: Path):
    """
    Test preparation of MNIST covariates (which should be empty).
    
    This test ensures that the prepare_data function correctly handles
    the case of MNIST covariates, which should result in empty datasets.
    """
    out_path = str(tmpdir / "mnist_covariates.h5")
    prepare_data(
        out_path=out_path,
        dataset="mnist",
        features_targets_covariates="covariates",
        variant="none",
        custom_datasets={}
    )

    with h5py.File(out_path, "r") as f:
        assert 'data' in f and 'mask' in f, "Missing data or mask for MNIST covariates"
        assert f['data'].shape == (0,), "Unexpected non-empty data for MNIST covariates"
        assert f['mask'].shape == (0,), "Unexpected non-empty mask for MNIST covariates"