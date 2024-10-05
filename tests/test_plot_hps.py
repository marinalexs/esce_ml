"""
test_plot_hps.py
================

This module contains unit tests for the plot function in the plot_hps module
of the ESCE workflow. It tests various scenarios for hyperparameter plotting.

Test Summary:
1. test_plot_hps_basic: Tests the basic functionality of plotting hyperparameters.
2. test_plot_hps_single_hyperparameter: Tests plotting with a single hyperparameter.
3. test_plot_hps_empty_input: Tests behavior when an empty input file is provided.
4. test_plot_hps_no_hyperparameters: Tests behavior when no hyperparameters are provided.
5. test_plot_hps_regression: Tests plotting for regression tasks (R² metric).
6. test_plot_hps_classification: Tests plotting for classification tasks (accuracy metric).
7. test_plot_hps_scale_handling: Tests correct application of linear and logarithmic scales.
8. test_plot_hps_reference_lines: Tests the presence of reference lines for hyperparameter ranges.
9. test_plot_hps_error_handling: Tests error handling for invalid input data.

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
import json

from workflow.scripts.plot_hps import plot


def test_plot_hps_basic(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test the basic functionality of plotting hyperparameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_scores_data(tmpdir_path)
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
            assert 'l1_ratio' in content, "Hyperparameter 'l1_ratio' not found in the output file"
            assert 'r2_val' in content, "R² metric not found in the output file"

def test_plot_hps_single_hyperparameter(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test plotting with a single hyperparameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_scores_data(tmpdir_path, n_hyperparameters=1)
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
            assert 'l1_ratio' not in content, "Hyperparameter 'l1_ratio' should not be in the output file"

def test_plot_hps_empty_input(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test behavior when an empty input file is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        empty_stats_file = tmpdir_path / "empty_stats.csv"
        empty_stats_file.touch()
        
        output_file = tmpdir_path / "test_plot_hps_empty.html"

        plot(
            stats_filename=str(empty_stats_file),
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test Empty Input Plot"
        )

        assert output_file.exists(), "Output file was not created"
        assert output_file.stat().st_size == 0, "Output file should be empty"

def test_plot_hps_no_hyperparameters(sample_hyperparameter_scales, generate_scores_data):
    """Test behavior when no hyperparameters are provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_scores_data(tmpdir_path)
        output_file = tmpdir_path / "test_plot_hps_no_hp.html"

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": {}},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test No Hyperparameters Plot"
        )

        assert output_file.exists(), "Output file was not created"
        assert output_file.stat().st_size == 0, "Output file should be empty"

def test_plot_hps_regression(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test plotting for regression tasks (R² metric)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_scores_data(tmpdir_path, task='regression')
        output_file = tmpdir_path / "test_plot_hps_regression.html"

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test Regression Plot"
        )

        assert output_file.exists(), "Output file was not created"
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'r2_val' in content, "R² metric not found in the output file"

def test_plot_hps_classification(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test plotting for classification tasks (accuracy metric)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_scores_data(tmpdir_path, task='classification')
        output_file = tmpdir_path / "test_plot_hps_classification.html"

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test Classification Plot"
        )

        assert output_file.exists(), "Output file was not created"
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'acc_val' in content, "Accuracy metric not found in the output file"

def test_plot_hps_scale_handling(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test correct application of linear and logarithmic scales."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_scores_data(tmpdir_path)
        output_file = tmpdir_path / "test_plot_hps_scales.html"
        json_file = output_file.with_suffix('.json')

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test Scale Handling Plot"
        )

        assert json_file.exists(), "JSON file was not created"
        with open(json_file, 'r') as f:
            content = json.load(f)
        
        # Check for logarithmic scale
        scales = [layer['encoding']['y']['scale']['type'] for chart in content['hconcat'] for layer in chart['layer'] if 'scale' in layer['encoding']['y']]
        assert 'log' in scales, "Logarithmic scale not found in the output file"
        assert 'linear' in scales, "Linear scale not found in the output file"

def test_plot_hps_reference_lines(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test the presence of reference lines for hyperparameter ranges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stats_file = generate_scores_data(tmpdir_path)
        output_file = tmpdir_path / "test_plot_hps_reference.html"
        json_file = output_file.with_suffix('.json')

        plot(
            stats_filename=stats_file,
            output_filename=str(output_file),
            grid={"TestModel": sample_grid},
            hyperparameter_scales=sample_hyperparameter_scales,
            model_name="TestModel",
            title="Test Reference Lines Plot"
        )

        assert json_file.exists(), "JSON file was not created"
        with open(json_file, 'r') as f:
            content = json.load(f)
        
        # Check for reference lines
        rule_marks = [layer['mark']['type'] for chart in content['hconcat'] for layer in chart['layer'] if layer['mark']['type'] == 'rule']
        assert rule_marks, "Reference lines not found in the output file"

def test_plot_hps_error_handling(sample_grid, sample_hyperparameter_scales, generate_scores_data):
    """Test error handling for invalid input data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        invalid_stats_file = tmpdir_path / "invalid_stats.csv"
        with open(invalid_stats_file, 'w') as f:
            f.write("invalid,data\n1,2\n")
        
        output_file = tmpdir_path / "test_plot_hps_error.html"

        with pytest.raises(Exception):  # Adjust the specific exception type as needed
            plot(
                stats_filename=str(invalid_stats_file),
                output_filename=str(output_file),
                grid={"TestModel": sample_grid},
                hyperparameter_scales=sample_hyperparameter_scales,
                model_name="TestModel",
                title="Test Error Handling Plot"
            )