"""
test_prepare_data.py
====================

This module contains comprehensive unit tests for the prepare_data function in the prepare_data module
of the ESCE workflow. It tests the functionality of preparing both custom and predefined datasets
with various input file types and data categories.

Test Summary:
1. test_prepare_data_custom_datasets: Tests the preparation of custom datasets for different
   input file types (CSV, TSV, NPY) and data categories (features, targets, covariates).
2. test_prepare_data_empty_covariates: Tests the handling of empty covariate datasets.
3. test_prepare_data_errors: Tests error handling for various edge cases (unsupported file formats,
   empty datasets, incompatible data types).
4. test_prepare_data_predefined_dataset: Tests the preparation of predefined datasets (MNIST features and targets).
5. test_prepare_data_missing_dataset: Tests error handling for non-existent datasets.
6. test_mnist_features: Tests the preparation of MNIST features.
7. test_mnist_targets: Tests the preparation of MNIST targets for different variants.
8. test_mnist_covariates: Tests the preparation of MNIST covariates (which should be empty).

These tests ensure that the prepare_data function correctly handles different input
formats, properly processes the data, and converts them to the required HDF5 format.
They also verify that appropriate error messages are raised for invalid inputs.
"""

import h5py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from workflow.scripts.prepare_data import prepare_data, predefined_datasets, FLOAT_PRECISION

@pytest.fixture
def custom_datasets(tmpdir):
    # Create dummy data
    features = np.random.rand(100, 10)
    targets = np.random.randint(0, 2, 100)
    covariates = np.random.rand(100, 5)

    # Save data in different formats
    formats = ['csv', 'tsv', 'npy']
    datasets = {}

    for fmt in formats:
        datasets[fmt] = {
            'features': {'normal': str(tmpdir / f"features.{fmt}")},
            'targets': {'normal': str(tmpdir / f"targets.{fmt}")},
            'covariates': {'normal': str(tmpdir / f"covariates.{fmt}")}
        }
        
        if fmt in ['csv', 'tsv']:
            sep = ',' if fmt == 'csv' else '\t'
            pd.DataFrame(features).to_csv(datasets[fmt]['features']['normal'], index=False, sep=sep)
            pd.DataFrame(targets).to_csv(datasets[fmt]['targets']['normal'], index=False, sep=sep)
            pd.DataFrame(covariates).to_csv(datasets[fmt]['covariates']['normal'], index=False, sep=sep)
        else:
            np.save(datasets[fmt]['features']['normal'], features)
            np.save(datasets[fmt]['targets']['normal'], targets)
            np.save(datasets[fmt]['covariates']['normal'], covariates)

    return datasets

def test_prepare_data_custom_datasets(tmpdir, custom_datasets):
    for fmt in custom_datasets:
        for data_type in ['features', 'targets', 'covariates']:
            out_path = str(tmpdir / f"{data_type}_{fmt}.h5")
            prepare_data(
                out_path=out_path,
                dataset='custom',
                features_targets_covariates=data_type,
                variant='normal',
                custom_datasets={'custom': custom_datasets[fmt]}
            )

            with h5py.File(out_path, 'r') as f:
                assert 'data' in f
                assert 'mask' in f
                assert f['data'].dtype == FLOAT_PRECISION
                assert f['mask'].dtype == bool

                if data_type == 'targets':
                    assert f['data'].ndim == 1
                elif data_type == 'features':
                    assert f['data'].shape[1] == 10
                elif data_type == 'covariates':
                    assert f['data'].shape[1] == 5

def test_prepare_data_empty_covariates(tmpdir):
    out_path = str(tmpdir / "empty_covariates.h5")
    prepare_data(
        out_path=out_path,
        dataset='custom',
        features_targets_covariates='covariates',
        variant='none',
        custom_datasets={}
    )

    with h5py.File(out_path, 'r') as f:
        assert 'data' in f
        assert 'mask' in f
        assert f['data'].shape == (0,)
        assert f['mask'].shape == (0,)

def test_prepare_data_errors(tmpdir, custom_datasets):
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

def test_prepare_data_predefined_dataset(tmpdir):
    out_path = str(tmpdir / "mnist_features.h5")
    prepare_data(
        out_path=out_path,
        dataset='mnist',
        features_targets_covariates='features',
        variant='pixel',
        custom_datasets={}
    )

    with h5py.File(out_path, 'r') as f:
        assert 'data' in f
        assert 'mask' in f
        assert f['data'].shape[1] == 784  # MNIST has 28x28=784 pixels

def test_prepare_data_missing_dataset(tmpdir):
    with pytest.raises(KeyError, match="Requested predefined dataset or variant does not exist"):
        prepare_data(
            out_path=str(tmpdir / "error.h5"),
            dataset='nonexistent',
            features_targets_covariates='features',
            variant='normal',
            custom_datasets={}
        )

@pytest.mark.slow
def test_mnist_features(tmpdir):
    out_path = str(tmpdir / "mnist_features.h5")
    prepare_data(
        out_path=out_path,
        dataset="mnist",
        features_targets_covariates="features",
        variant="pixel",
        custom_datasets={}
    )

    with h5py.File(out_path, "r") as f:
        assert 'data' in f
        assert 'mask' in f
        assert f['data'].dtype == FLOAT_PRECISION
        assert f['mask'].dtype == bool
        assert f['data'].shape == (70000, 784)  # 70,000 samples, 28x28 pixels
        assert f['mask'].shape == (70000,)
        data_array = f['data'][:]  # Load the entire dataset into memory
        assert np.all(data_array >= 0) and np.all(data_array <= 255)  # Pixel values are 0-255

@pytest.mark.slow
def test_mnist_targets(tmpdir):
    for variant in ['ten-digits', 'odd-even']:
        out_path = str(tmpdir / f"mnist_targets_{variant}.h5")
        prepare_data(
            out_path=out_path,
            dataset="mnist",
            features_targets_covariates="targets",
            variant=variant,
            custom_datasets={}
        )

        with h5py.File(out_path, "r") as f:
            assert 'data' in f
            assert 'mask' in f
            assert f['data'].dtype == FLOAT_PRECISION
            assert f['mask'].dtype == bool
            assert f['data'].shape == (70000,)  # 70,000 samples
            assert f['mask'].shape == (70000,)

            if variant == 'ten-digits':
                assert set(np.unique(f['data'])) == set(range(10))
            elif variant == 'odd-even':
                assert set(np.unique(f['data'])) == {0, 1}

def test_mnist_covariates(tmpdir):
    out_path = str(tmpdir / "mnist_covariates.h5")
    prepare_data(
        out_path=out_path,
        dataset="mnist",
        features_targets_covariates="covariates",
        variant="none",
        custom_datasets={}
    )

    with h5py.File(out_path, "r") as f:
        assert 'data' in f
        assert 'mask' in f
        assert f['data'].shape == (0,)
        assert f['mask'].shape == (0,)
