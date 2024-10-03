"""
test_aggregate.py
====================================

This module contains unit tests for the aggregate function in the aggregate.py script.
It tests the functionality of aggregating scores from multiple files and selecting
the best hyperparameter combinations.

The tests cover the following scenarios:
1. Aggregating R² metric scores
2. Handling empty input files
3. Aggregating accuracy metric scores

"""

import pandas as pd
import pytest
from pathlib import Path

from workflow.scripts.aggregate import aggregate


@pytest.fixture
def sample_data_r2(tmpdir):
    """
    Fixture to create sample data with R² metric for testing.

    Args:
        tmpdir: Pytest fixture for creating a temporary directory.

    Returns:
        tuple: A tuple containing the list of score file paths and the stats output path.
    """
    scores1 = {"n": [100, 200], "s": [42, 43], "r2_val": [0.8, 0.9], "other_param": ["a", "b"]}
    scores2 = {"n": [100, 200], "s": [42, 43], "r2_val": [0.7, 0.85], "other_param": ["c", "d"]}

    df1 = pd.DataFrame(scores1)
    df2 = pd.DataFrame(scores2)

    scores_path1 = tmpdir.join("scores1.csv")
    scores_path2 = tmpdir.join("scores2.csv")

    df1.to_csv(scores_path1, index=False)
    df2.to_csv(scores_path2, index=False)

    score_path_list = [str(scores_path1), str(scores_path2)]
    stats_path = str(tmpdir.join("stats.csv"))

    return score_path_list, stats_path


def test_aggregate_r2(sample_data_r2):
    """
    Test the aggregate function with R² metric.

    This test checks if the aggregate function correctly identifies and saves
    the best hyperparameter combinations based on the highest R² values.

    Args:
        sample_data_r2: Pytest fixture providing sample data for testing.
    """
    score_path_list, stats_path = sample_data_r2

    # Run the aggregate function
    aggregate(score_path_list, stats_path)

    # Load the stats and check if they are correct
    stats = pd.read_csv(stats_path)

    # Check if the expected columns are present
    assert set(stats.columns) == {"n", "s", "r2_val", "other_param"}, "Unexpected columns in the output"

    # Check if the number of rows is correct (one for each unique n-s combination)
    assert len(stats) == 2, "Incorrect number of rows in the output"

    # Check if the best R² values were selected for each n-s combination
    assert stats.loc[stats["n"] == 100, "r2_val"].values[0] == 0.8, "Incorrect R² value selected for n=100"
    assert stats.loc[stats["n"] == 200, "r2_val"].values[0] == 0.9, "Incorrect R² value selected for n=200"

    # Check if the corresponding other_param values were correctly selected
    assert stats.loc[stats["n"] == 100, "other_param"].values[0] == "a", "Incorrect other_param value selected for n=100"
    assert stats.loc[stats["n"] == 200, "other_param"].values[0] == "b", "Incorrect other_param value selected for n=200"


def test_aggregate_empty_files(tmpdir):
    """
    Test the aggregate function with empty input files.

    This test checks if the aggregate function correctly handles the case
    when all input files are empty.

    Args:
        tmpdir: Pytest fixture for creating a temporary directory.
    """
    # Create empty score files
    empty_file1 = tmpdir.join("empty1.csv")
    empty_file2 = tmpdir.join("empty2.csv")
    Path(empty_file1).touch()
    Path(empty_file2).touch()

    score_path_list = [str(empty_file1), str(empty_file2)]
    stats_path = str(tmpdir.join("stats.csv"))

    # Run the aggregate function
    aggregate(score_path_list, stats_path)

    # Check if the output file is created but empty
    assert Path(stats_path).exists(), "Output file was not created"
    assert Path(stats_path).stat().st_size == 0, "Output file is not empty"


def test_aggregate_accuracy(tmpdir):
    """
    Test the aggregate function with accuracy metric.

    This test checks if the aggregate function correctly handles input files
    with the accuracy metric.

    Args:
        tmpdir: Pytest fixture for creating a temporary directory.
    """
    scores1 = {"n": [100, 200], "s": [42, 43], "acc_val": [0.7, 0.85]}
    scores2 = {"n": [100, 200], "s": [42, 43], "acc_val": [0.75, 0.8]}

    df1 = pd.DataFrame(scores1)
    df2 = pd.DataFrame(scores2)

    scores_path1 = tmpdir.join("scores1.csv")
    scores_path2 = tmpdir.join("scores2.csv")

    df1.to_csv(scores_path1, index=False)
    df2.to_csv(scores_path2, index=False)

    score_path_list = [str(scores_path1), str(scores_path2)]
    stats_path = str(tmpdir.join("stats.csv"))

    # Run the aggregate function
    aggregate(score_path_list, stats_path)

    # Load the stats and check if they are correct
    stats = pd.read_csv(stats_path)

    # Check if the function correctly used acc_val as the metric
    assert "acc_val" in stats.columns, "acc_val column is missing in the output"
    assert "r2_val" not in stats.columns, "r2_val column should not be present in the output"

    # Check if the best accuracy values were selected
    assert stats.loc[stats["n"] == 100, "acc_val"].values[0] == 0.75, "Incorrect accuracy value selected for n=100"
    assert stats.loc[stats["n"] == 200, "acc_val"].values[0] == 0.85, "Incorrect accuracy value selected for n=200"