import pytest
import numpy as np
import pandas as pd
import os
from esce.predefined_datasets import predefined_datasets
from esce.prepare_data import prepare_data


@pytest.mark.parametrize('out_path, dataset, features_targets_covariates, variant, custom_datasets, dummy_data, expected_output', [
    ('test.npy', 'custom_dataset', 'features', 'variant1', {
        'custom_dataset': {
            'features': {
                'variant1': 'data.csv'
            }
        }
    },[[1, 2], [3, 4]],[[1, 2], [3, 4]]),
    ('test.npy', 'custom_dataset', 'targets', 'variant2', {
        'custom_dataset': {
            'targets': {
                'variant2': 'data.npy'
            }
        }
    },[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],),
    ('test.npy', 'custom_dataset', 'covariates', 'variant3', {
        'custom_dataset': {
            'covariates': {
                'variant3': 'data.tsv'
            }
        }
    },[[1, 2], [3, 4]], [[1, 2], [3, 4]]),
    ('test.npy', 'custom_dataset', 'covariates', 'variant4', {
        'custom_dataset': {
            'covariates': {
                'variant4': 'data.tsv'
            }
        }
    },[1, 2, 3, 4, 5], [[1,], [2,], [3,], [4,], [5,]]),
])
def test_prepare_data_custom_datasets(out_path, dataset, features_targets_covariates, variant, custom_datasets, dummy_data, expected_output):
    in_path = custom_datasets[dataset][features_targets_covariates][variant]
    dummy_data = pd.DataFrame(dummy_data)
    expected_output = np.array(expected_output)

    # Create some test data to simulate the custom dataset
    if in_path.endswith('.csv'):
        dummy_data.to_csv(in_path, index=False)
    elif in_path.endswith('.tsv'):
        dummy_data.to_csv(in_path, sep='\t', index=False)
    elif in_path.endswith('.npy'):
        np.save(in_path, dummy_data.values)

    prepare_data(out_path, dataset, features_targets_covariates, variant, custom_datasets)

    # Check if the output file was created
    assert os.path.exists(out_path)

    # Load the data from the output file and check its dimensions
    output_data = np.load(out_path)
    assert output_data.ndim == expected_output.ndim
    assert np.allclose(output_data, expected_output)

    # Clean up
    os.remove(in_path)
    os.remove(out_path)
