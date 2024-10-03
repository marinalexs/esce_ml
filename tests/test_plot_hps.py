"""
test_plot_hps.py
================

This module contains unit tests for the plot function in the plot_hps module
of the ESCE workflow. It tests various scenarios for hyperparameter plotting.

Test Summary:
1. test_plot_hps: Tests the basic functionality of plotting hyperparameters.
2. test_plot_hps_single_hyperparameter: Tests plotting with a single hyperparameter.
3. test_plot_hps_no_data: Tests behavior when no data is provided.
4. test_plot_hps_invalid_input: Tests error handling for invalid input.
5. test_plot_hps_different_metrics: Tests plotting with different performance metrics.

These tests ensure that the hyperparameter plotting functionality works correctly
under various conditions and handles edge cases appropriately.
"""

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import altair as alt

from workflow.scripts.plot_hps import plot

def generate_sample_data(path, n_samples=5, n_hyperparameters=2):
    """
    Generate synthetic data for testing.

    Args:
        path (Path): Directory path to store the generated data.
        n_samples (int): Number of samples to generate.
        n_hyperparameters (int): Number of hyperparameters to generate.

    Returns:
        str: Path to the generated stats file.
    """
    data = []
    hyperparameters = ['alpha', 'beta'][:n_hyperparameters]
    
    for i in range(n_samples):
        row = {
            'n': 2 ** (7 + i),  # 128, 256, 512, 1024, 2048
            's': np.random.randint(1, 6),
            'r2_val': np.random.uniform(0.5, 0.9),
            'acc_val': np.random.uniform(0.6, 0.95)
        }
        for hp in hyperparameters:
            row[hp] = np.random.uniform(0.01, 1.0)
        data.append(row)
    
    df = pd.DataFrame(data)
    stats_file = path / "stats.csv"
    df.to_csv(stats_file, index=False)
    
    return str(stats_file)

@pytest.fixture
def sample_grid():
    """Fixture for sample hyperparameter grid."""
    return {'alpha': [0.1, 0.5, 1.0], 'beta': [0.01, 0.1, 1.0]}

@pytest.fixture
def sample_hyperparameter_scales():
    """Fixture for sample hyperparameter scales."""
    return {'alpha': 'log', 'beta': 'linear'}

def test_plot_hps(sample_grid, sample_hyperparameter_scales):
    """
    Test the basic functionality of plotting hyperparameters.

    Args:
        sample_grid (dict): Sample hyperparameter grid.
        sample_hyperparameter_scales (dict): Sample hyperparameter scales.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_sample_data(tmpdir_path)
        output_file = tmpdir_path / "test_plot_hps.html"

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test Hyperparameter Plot"
        )

        assert output_file.exists(), "Output file was not created"
        assert output_file.stat().st_size > 0, "Output file is empty"

        with open(output_file, 'r') as f:
            content = f.read()
            assert 'Test Hyperparameter Plot' in content, "Title not found in the output file"
            assert 'vega-embed' in content, "Vega-Embed not found in the output file"
            assert 'alpha' in content, "Hyperparameter 'alpha' not found in the output file"
            assert 'beta' in content, "Hyperparameter 'beta' not found in the output file"
            assert 'r2_val' in content or 'acc_val' in content, "Neither R2 nor Accuracy metric found in the output file"

def test_plot_hps_single_hyperparameter(sample_grid, sample_hyperparameter_scales):
    """
    Test plotting with a single hyperparameter.

    Args:
        sample_grid (dict): Sample hyperparameter grid.
        sample_hyperparameter_scales (dict): Sample hyperparameter scales.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_sample_data(tmpdir_path, n_hyperparameters=1)
        output_file = tmpdir_path / "test_plot_hps_single.html"

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": {'alpha': sample_grid['alpha']}},
            hyperparameter_scales={'alpha': sample_hyperparameter_scales['alpha']},
            model_name="TestModel",
            title="Test Single Hyperparameter Plot"
        )

        assert output_file.exists(), "Output file was not created"
        assert output_file.stat().st_size > 0, "Output file is empty"

        with open(output_file, 'r') as f:
            content = f.read()
            assert 'alpha' in content, "Hyperparameter 'alpha' not found in the output file"
            assert 'beta' not in content, "Hyperparameter 'beta' should not be in the output file"
            assert 'r2_val' in content or 'acc_val' in content, "Neither R2 nor Accuracy metric found in the output file"

def test_plot_hps_no_data(sample_grid, sample_hyperparameter_scales):
    """
    Test behavior when no data is provided.

    Args:
        sample_grid (dict): Sample hyperparameter grid.
        sample_hyperparameter_scales (dict): Sample hyperparameter scales.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        empty_stats_file = tmpdir_path / "empty_stats.csv"
        pd.DataFrame(columns=['n', 'r2_val', 'acc_val', 'alpha', 'beta']).to_csv(empty_stats_file, index=False)
        
        output_file = tmpdir_path / "test_plot_hps_no_data.html"

        plot(
            stats_filename=str(empty_stats_file),
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test No Data Plot"
        )

        assert output_file.exists(), "Output file was not created"
        assert output_file.stat().st_size == 0, "Output file should be empty"

def test_plot_hps_invalid_input(sample_grid, sample_hyperparameter_scales):
    """
    Test error handling for invalid input.

    Args:
        sample_grid (dict): Sample hyperparameter grid.
        sample_hyperparameter_scales (dict): Sample hyperparameter scales.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        non_existent_file = tmpdir_path / "non_existent.csv"
        output_file = tmpdir_path / "test_plot_hps_invalid.html"

        with pytest.raises(FileNotFoundError):
            plot(
                stats_filename=str(non_existent_file),
                output_filename=str(output_file),
                grid={"TestModel": sample_grid},
                hyperparameter_scales=sample_hyperparameter_scales,
                model_name="TestModel",
                title="Test Invalid Input Plot"
            )

def test_plot_hps_different_metrics(sample_grid, sample_hyperparameter_scales):
    """
    Test plotting with different performance metrics.

    Args:
        sample_grid (dict): Sample hyperparameter grid.
        sample_hyperparameter_scales (dict): Sample hyperparameter scales.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Generate data with r2_val and acc_val
        stats_file = generate_sample_data(tmpdir_path)
        
        output_file = tmpdir_path / "test_plot_hps_metrics.html"

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test Metrics Plot"
        )

        assert output_file.exists(), "Output file should be created"
        
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'r2_val' in content or 'acc_val' in content, "Neither R2 nor Accuracy metric found in the plot file"

# Add more tests as needed for other functions or edge cases in plot_hps.py