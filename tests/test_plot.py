"""
test_plot.py
============

This module contains unit tests for the plot and process_results functions
in the plot module of the ESCE workflow.

Test Summary:
1. test_plot: Tests the plot function by generating synthetic data and verifying the output.
2. test_process_results_invalid_structure: Tests that process_results raises an error for invalid file structures.
3. test_process_results_invalid_filename: Tests that process_results raises an error for invalid filenames.
4. test_process_results: Tests the process_results function by checking the structure and content
   of the resulting DataFrame.
5. test_empty_result_files: Tests handling of empty result files.
6. test_missing_bootstrap_files: Tests behavior when bootstrap files are missing.
7. test_invalid_bootstrap_data: Tests error handling for invalid bootstrap data.

These tests ensure that the plotting functionality works correctly, results are processed as expected,
and edge cases are handled properly.
"""

import pandas as pd
import pytest
import numpy as np
from pathlib import Path
import json

from workflow.scripts.plot import plot, process_results

def test_plot(generate_stats_data, write_stats_data, construct_filename, tmp_path):
    """
    Test the plot function by generating synthetic data and verifying the output.
    """
    # Generate synthetic data
    n_samples = 5
    sample_results = []
    for i in range(n_samples):
        x, y, y_err, bootstrap_params = generate_stats_data(f"dataset{i+1}", random_state=i, n_bootstrap=3)
        
        # Create a more realistic directory structure
        dataset = f"dataset{i+1}"
        model = "model1"
        features = "features1"
        target = "target1"
        confound_correction_method = "correct-x"
        confound_correction_cni = "cni1"
        balanced = "balanced"
        grid = "grid1"
        
        base_path = tmp_path / "results" / dataset / "statistics" / model
        base_path.mkdir(parents=True, exist_ok=True)
        
        filename = construct_filename({
            'features': features,
            'target': target,
            'confound_correction_method': confound_correction_method,
            'confound_correction_cni': confound_correction_cni,
            'balanced': balanced,
            'grid': grid
        })
        full_path = base_path / filename
        
        stats_file, bootstrap_file = write_stats_data(str(full_path), x, y, y_err, bootstrap_params)
        sample_results.append(stats_file)

    # Define output file
    output_file = tmp_path / "test_plot.html"

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

def test_process_results_invalid_structure(tmp_path):
    """
    Test that process_results raises an error for invalid file structures.
    """
    # Create an invalid file structure
    invalid_file = tmp_path / "invalid" / "file.stats.json"
    invalid_file.parent.mkdir(parents=True, exist_ok=True)
    invalid_file.touch()

    with pytest.raises(ValueError, match="Invalid filename structure:.*"):
        process_results([str(invalid_file)])

def test_process_results_invalid_filename(tmp_path):
    """
    Test that process_results raises an error for invalid filenames.
    """
    # Create a file with an invalid filename
    invalid_file = tmp_path / "results" / "dataset1" / "statistics" / "model1" / "invalid_filename.stats.json"
    invalid_file.parent.mkdir(parents=True, exist_ok=True)
    invalid_file.touch()

    with pytest.raises(ValueError, match="Invalid filename structure:.*"):
        process_results([str(invalid_file)])

def test_process_results(generate_stats_data, write_stats_data, parse_filename, construct_filename, tmp_path):
    """
    Test the process_results function by checking the structure and content of the resulting DataFrame.
    """
    # Generate synthetic data
    n_samples = 5
    sample_results = []
    for i in range(n_samples):
        x, y, y_err, bootstrap_params = generate_stats_data(f"dataset{i+1}", random_state=i, n_bootstrap=3)
        
        # Create a more realistic directory structure
        dataset = f"dataset{i+1}"
        model = "model1"
        features = "features1"
        target = "target1"
        confound_correction_method = "correct-x"
        confound_correction_cni = "cni1"
        balanced = "balanced"
        grid = "grid1"
        
        base_path = tmp_path / "results" / dataset / "statistics" / model
        base_path.mkdir(parents=True, exist_ok=True)
        
        filename = construct_filename({
            'features': features,
            'target': target,
            'confound_correction_method': confound_correction_method,
            'confound_correction_cni': confound_correction_cni,
            'balanced': balanced,
            'grid': grid
        })
        full_path = base_path / filename
        
        stats_file, _ = write_stats_data(str(full_path), x, y, y_err, bootstrap_params)
        sample_results.append(stats_file)

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
        path_parts = Path(row['full_path']).parts
        assert row['dataset'] == path_parts[-6], f"Mismatch in dataset for row {index}"
        assert row['model'] == path_parts[-4], f"Mismatch in model for row {index}"
        
        filename = Path(row['full_path']).stem.replace('.stats', '')
        parsed = parse_filename(filename)
        for key, value in parsed.items():
            assert row[key] == value, f"Mismatch in {key} for row {index}. Expected {value}, got {row[key]}"

    # Check if the 'cni' column is created correctly
    for index, row in df.iterrows():
        expected_cni = f"{row['confound_correction_method']}-{row['confound_correction_cni']}"
        assert row['cni'] == expected_cni, f"Mismatch in cni for row {index}. Expected {expected_cni}, got {row['cni']}"

def test_empty_result_files(tmp_path, construct_filename):
    """
    Test handling of empty result files.
    """
    # Create an empty stats file
    # Use the construct_filename fixture to create the filenames
    empty_stats_filename = construct_filename({
        'features': 'features1',
        'target': 'target1',
        'confound_correction_method': 'correct-x',
        'confound_correction_cni': 'cni1',
        'balanced': 'balanced',
        'grid': 'grid1'
    })
    empty_stats_file = tmp_path / "results" / "dataset1" / "statistics" / "model1" / f"{empty_stats_filename}_stats.json"
    empty_stats_file.parent.mkdir(parents=True, exist_ok=True)
    empty_stats_file.touch()
    
    # Create a non-empty stats file
    non_empty_stats_filename = construct_filename({
        'features': 'features1',
        'target': 'target1',
        'confound_correction_method': 'correct-x',
        'confound_correction_cni': 'cni1',
        'balanced': 'balanced',
        'grid': 'grid1'
    })
    non_empty_stats_file = tmp_path / "results" / "dataset2" / "statistics" / "model1" / f"{non_empty_stats_filename}_stats.json"
    non_empty_stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(non_empty_stats_file, 'w') as f:
        json.dump({"x": [100], "y_mean": [0.5], "y_std": [0.1]}, f)
    
    # Process the results
    df = process_results([str(empty_stats_file), str(non_empty_stats_file)])
    
    # Check that only the non-empty file is processed
    assert len(df) == 1, "Only non-empty files should be processed"
    assert df.iloc[0]['full_path'] == str(non_empty_stats_file), "Non-empty file should be processed"

def test_missing_bootstrap_files(generate_stats_data, write_stats_data, construct_filename, tmp_path):
    """
    Test behavior when bootstrap files are missing.
    """
    x, y, y_err, bootstrap_params = generate_stats_data("dataset1", random_state=42, n_bootstrap=3)
    
    base_path = tmp_path / "results" / "dataset1" / "statistics" / "model1"
    base_path.mkdir(parents=True, exist_ok=True)
    
    filename = construct_filename({
        'features': "features1",
        'target': "target1",
        'confound_correction_method': "correct-x",
        'confound_correction_cni': "cni1",
        'balanced': "balanced",
        'grid': "grid1"
    })
    full_path = base_path / filename
    
    stats_file, bootstrap_file = write_stats_data(str(full_path), x, y, y_err, bootstrap_params)
    
    # Remove the bootstrap file
    Path(bootstrap_file).unlink()
    
    # Define output file
    output_file = tmp_path / "test_plot.html"
    
    # Call the plot function
    plot(
        stats_file_list=[stats_file],
        output_filename=str(output_file),
        color_variable="dataset",
        linestyle_variable=None,
        title="Test Plot",
        max_x=6
    )
    
    # Check if the output file was created despite missing bootstrap file
    assert output_file.exists(), "Output file should be created even with missing bootstrap file"

def test_invalid_bootstrap_data(generate_stats_data, write_stats_data, construct_filename, tmp_path):
    """
    Test error handling for invalid bootstrap data.
    """
    x, y, y_err, _ = generate_stats_data("dataset1", random_state=42, n_bootstrap=3)
    
    base_path = tmp_path / "results" / "dataset1" / "statistics" / "model1"
    base_path.mkdir(parents=True, exist_ok=True)
    
    filename = construct_filename({
        'features': "features1",
        'target': "target1",
        'confound_correction_method': "correct-x",
        'confound_correction_cni': "cni1",
        'balanced': "balanced",
        'grid': "grid1"
    })
    full_path = base_path / filename
    
    # Create invalid bootstrap data
    invalid_bootstrap_data = [["invalid", "data"]]
    
    stats_file, bootstrap_file = write_stats_data(str(full_path), x, y, y_err, np.array(invalid_bootstrap_data))
    
    # Define output file
    output_file = tmp_path / "test_plot.html"
    
    # Call the plot function
    plot(
        stats_file_list=[stats_file],
        output_filename=str(output_file),
        color_variable="dataset",
        linestyle_variable=None,
        title="Test Plot",
        max_x=6
    )
    
    # Check if the output file was created despite invalid bootstrap data
    assert output_file.exists(), "Output file should be created even with invalid bootstrap data"