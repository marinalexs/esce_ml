"""
test_plot.py
============

This module contains unit tests for the plot and process_results functions
in the plot module of the ESCE workflow.

Test Summary:
1. test_plot: Tests the plot function by generating synthetic data and verifying the output.
2. test_process_results: Tests the process_results function by checking the structure and content
   of the resulting DataFrame.
3. test_empty_result_files: Tests handling of empty result files.
4. test_missing_bootstrap_files: Tests behavior when bootstrap files are missing.
5. test_invalid_bootstrap_data: Tests error handling for invalid bootstrap data.

These tests ensure that the plotting functionality works correctly, results are processed as expected,
and edge cases are handled properly.
"""

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import os
import yaml

from workflow.scripts.plot import plot, process_results

def generate_sample_data(path, n_samples=5):
    """
    Generate synthetic data for testing.

    Args:
        path (Path): Directory path to store the generated data.
        n_samples (int): Number of samples to generate.

    Returns:
        list: List of paths to the generated stats files.
    """
    data = []
    for i in range(n_samples):
        sample_size = 2 ** (7 + i)  # 128, 256, 512, 1024, 2048
        stats = {
            "x": [sample_size],
            "y_mean": [np.random.uniform(0.5, 0.9)],
            "y_std": [np.random.uniform(0.01, 0.1)]
        }
        bootstrap = [[np.random.uniform(0.1, 1), np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)] for _ in range(3)]
        
        base_path = path / f"dataset{i+1}_model1_features1_target1_correct-x_cni1_balanced_grid1/statistics"
        base_path.mkdir(parents=True, exist_ok=True)
        
        stats_file = base_path / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        
        bootstrap_file = base_path / "bootstrap.json"
        with open(bootstrap_file, 'w') as f:
            json.dump(bootstrap, f)
        
        data.append(str(stats_file))
    return data

def test_plot():
    """
    Test the plot function by generating synthetic data and verifying the output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic data
        tmpdir_path = Path(tmpdir)
        sample_results = generate_sample_data(tmpdir_path)

        # Define output file
        output_file = tmpdir_path / "test_plot.html"

        # Call the plot function
        plot(
            stats_file_list=sample_results,
            output_filename=str(output_file),
            color_variable="dataset",
            linestyle_variable=None,
            title="Test Plot",
            max_x=6
        )

        # Check if the output file was created
        assert output_file.exists(), "Output file was not created"

        # Check if the file is not empty
        assert output_file.stat().st_size > 0, "Output file is empty"

        # Check if the file contains some expected content
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'Test Plot' in content, "Title not found in the output file"
            assert 'vega-embed' in content, "Vega-Embed not found in the output file"

def test_process_results():
    """
    Test the process_results function by checking the structure and content of the resulting DataFrame.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic data
        tmpdir_path = Path(tmpdir)
        sample_results = generate_sample_data(tmpdir_path)

        # Process the results
        df = process_results(sample_results)

        # Check the DataFrame
        assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
        assert len(df) == len(sample_results), "DataFrame should have the same number of rows as input files"
        
        expected_columns = {
            'full_path', 'dataset', 'model', 'features', 'target',
            'confound_correction_method', 'confound_correction_cni',
            'balanced', 'grid', 'cni'
        }
        assert set(df.columns) == expected_columns, f"Unexpected columns: {set(df.columns) - expected_columns}"

        # Check if the columns are correctly extracted
        for index, row in df.iterrows():
            path_parts = Path(row['full_path']).parts[-3].split('_')
            assert row['dataset'] == path_parts[0], f"Mismatch in dataset for row {index}"
            assert row['model'] == path_parts[1], f"Mismatch in model for row {index}"
            assert row['features'] == path_parts[2], f"Mismatch in features for row {index}"
            assert row['target'] == path_parts[3], f"Mismatch in target for row {index}"
            assert row['confound_correction_method'] == path_parts[4], f"Mismatch in confound_correction_method for row {index}"
            assert row['confound_correction_cni'] == path_parts[5], f"Mismatch in confound_correction_cni for row {index}"
            assert row['balanced'] == path_parts[6], f"Mismatch in balanced for row {index}"
            assert row['grid'] == path_parts[7], f"Mismatch in grid for row {index}"

        # Check if the 'cni' column is created correctly
        for index, row in df.iterrows():
            expected_cni = f"{row['confound_correction_method']}-{row['confound_correction_cni']}"
            assert row['cni'] == expected_cni, f"Mismatch in cni for row {index}. Expected {expected_cni}, got {row['cni']}"

def test_empty_result_files():
    """
    Test handling of empty result files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create an empty stats file
        empty_stats_file = tmpdir_path / "empty_stats.json"
        empty_stats_file.touch()
        
        # Create a non-empty stats file
        non_empty_stats_file = tmpdir_path / "non_empty_stats.json"
        with open(non_empty_stats_file, 'w') as f:
            json.dump({"x": [100], "y_mean": [0.5], "y_std": [0.1]}, f)
        
        # Process the results
        df = process_results([str(empty_stats_file), str(non_empty_stats_file)])
        
        # Check that only the non-empty file is processed
        assert len(df) == 1, "Only non-empty files should be processed"
        assert df.iloc[0]['full_path'] == str(non_empty_stats_file), "Non-empty file should be processed"

def test_missing_bootstrap_files():
    """
    Test behavior when bootstrap files are missing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        sample_results = generate_sample_data(tmpdir_path, n_samples=1)
        
        # Remove the bootstrap file
        bootstrap_file = Path(sample_results[0]).parent / "bootstrap.json"
        bootstrap_file.unlink()
        
        # Define output file
        output_file = tmpdir_path / "test_plot.html"
        
        # Call the plot function
        plot(
            stats_file_list=sample_results,
            output_filename=str(output_file),
            color_variable="dataset",
            linestyle_variable=None,
            title="Test Plot",
            max_x=6
        )
        
        # Check if the output file was created despite missing bootstrap file
        assert output_file.exists(), "Output file should be created even with missing bootstrap file"

def test_invalid_bootstrap_data():
    """
    Test error handling for invalid bootstrap data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        sample_results = generate_sample_data(tmpdir_path, n_samples=1)
        
        # Create invalid bootstrap data
        invalid_bootstrap_data = [["invalid", "data"]]
        bootstrap_file = Path(sample_results[0]).parent / "bootstrap.json"
        with open(bootstrap_file, 'w') as f:
            json.dump(invalid_bootstrap_data, f)
        
        # Define output file
        output_file = tmpdir_path / "test_plot.html"
        
        # Call the plot function
        plot(
            stats_file_list=sample_results,
            output_filename=str(output_file),
            color_variable="dataset",
            linestyle_variable=None,
            title="Test Plot",
            max_x=6
        )
        
        # Check if the output file was created despite invalid bootstrap data
        assert output_file.exists(), "Output file should be created even with invalid bootstrap data"
