import pytest
import numpy as np
import os
from esce.predefined_datasets import predefined_datasets
from esce.prepare_data import prepare_data


@pytest.mark.slow
def test_mnist_features():
    out_path = 'test.npy'
    dataset = 'mnist'
    features_targets_covariates = 'features'
    variant = 'pixel'
    custom_datasets = {}

    prepare_data(out_path, dataset, features_targets_covariates, variant, custom_datasets)

    # Check if the output file was created
    assert os.path.exists(out_path)

    # Load the data from the output file and check dimensions
    output_data = np.load(out_path)
    assert output_data.shape == (70000, 784)

    # Clean up
    os.remove(out_path)

@pytest.mark.slow
def test_mnist_targets():
    out_path = 'test.npy'
    dataset = 'mnist'
    features_targets_covariates = 'targets'
    variant = 'ten-digits'
    custom_datasets = {}

    prepare_data(out_path, dataset, features_targets_covariates, variant, custom_datasets)

    # Check if the output file was created
    assert os.path.exists(out_path)

    # Load the data from the output file and check dimensions
    output_data = np.load(out_path)
    assert output_data.shape == (70000, )

    # Clean up
    os.remove(out_path)
