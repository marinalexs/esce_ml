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
4. Handling mixed metrics (R² and accuracy)
5. Handling duplicate maximum scores
6. Testing with varying numbers of input files
7. Testing reproducibility
"""

import pandas as pd
import pytest
from pathlib import Path
from typing import List, Tuple, Callable
import os

from workflow.scripts.aggregate import aggregate

class TestAggregate:
    def test_aggregate_r2(self, tmpdir: Path, write_data_to_file: Callable) -> None:
        """
        Test the aggregate function with R² metric.

        This test verifies that:
        1. The output contains the expected columns
        2. The number of rows in the output is correct
        3. The correct R² values are selected for each sample size
        4. The correct other_param values are selected for each sample size

        Args:
            tmpdir: Pytest fixture for temporary directory.
            write_data_to_file: Fixture to write data to a file.
        """
        scores1 = {"n": [100, 200], "s": [42, 43], "r2_val": [0.8, 0.9], "other_param": ["a", "b"]}
        scores2 = {"n": [100, 200], "s": [42, 43], "r2_val": [0.7, 0.85], "other_param": ["c", "d"]}

        scores_path1 = write_data_to_file(pd.DataFrame(scores1), 'csv', tmpdir / "scores1.csv")
        scores_path2 = write_data_to_file(pd.DataFrame(scores2), 'csv', tmpdir / "scores2.csv")

        score_path_list = [str(scores_path1), str(scores_path2)]
        stats_path = str(tmpdir / "stats.csv")

        aggregate(score_path_list, stats_path)

        stats = pd.read_csv(stats_path)

        assert set(stats.columns) == {"n", "s", "r2_val", "other_param"}, "Unexpected columns in the output"
        assert len(stats) == 2, "Incorrect number of rows in the output"
        assert stats.loc[stats["n"] == 100, "r2_val"].values[0] == 0.8, "Incorrect R² value selected for n=100"
        assert stats.loc[stats["n"] == 200, "r2_val"].values[0] == 0.9, "Incorrect R² value selected for n=200"
        assert stats.loc[stats["n"] == 100, "other_param"].values[0] == "a", "Incorrect other_param value selected for n=100"
        assert stats.loc[stats["n"] == 200, "other_param"].values[0] == "b", "Incorrect other_param value selected for n=200"

    def test_aggregate_empty_files(self, tmpdir: Path) -> None:
        """
        Test the aggregate function with empty input files.

        This test verifies that:
        1. The output file is created
        2. The output file is empty when all input files are empty

        Args:
            tmpdir: Pytest fixture for temporary directory.
        """
        empty_file1 = Path(tmpdir) / "empty1.csv"
        empty_file2 = Path(tmpdir) / "empty2.csv"
        empty_file1.touch()
        empty_file2.touch()

        score_path_list = [str(empty_file1), str(empty_file2)]
        stats_path = str(Path(tmpdir) / "stats.csv")

        aggregate(score_path_list, stats_path)

        assert os.path.exists(stats_path), "Output file was not created"
        assert os.path.getsize(stats_path) == 0, "Output file is not empty"

    def test_aggregate_accuracy(self, tmpdir: Path, write_data_to_file: Callable) -> None:
        """
        Test the aggregate function with accuracy metric.

        This test verifies that:
        1. The output contains the accuracy column
        2. The output does not contain the R² column
        3. The correct accuracy values are selected for each sample size

        Args:
            tmpdir: Pytest fixture for temporary directory.
            write_data_to_file: Fixture to write data to a file.
        """
        scores1 = {"n": [100, 200], "s": [42, 43], "acc_val": [0.7, 0.85]}
        scores2 = {"n": [100, 200], "s": [42, 43], "acc_val": [0.75, 0.8]}

        scores_path1 = write_data_to_file(pd.DataFrame(scores1), 'csv', tmpdir / "scores1.csv")
        scores_path2 = write_data_to_file(pd.DataFrame(scores2), 'csv', tmpdir / "scores2.csv")

        score_path_list = [str(scores_path1), str(scores_path2)]
        stats_path = str(tmpdir / "stats.csv")

        aggregate(score_path_list, stats_path)

        stats = pd.read_csv(stats_path)

        assert "acc_val" in stats.columns, "acc_val column is missing in the output"
        assert "r2_val" not in stats.columns, "r2_val column should not be present in the output"
        assert stats.loc[stats["n"] == 100, "acc_val"].values[0] == 0.75, "Incorrect accuracy value selected for n=100"
        assert stats.loc[stats["n"] == 200, "acc_val"].values[0] == 0.85, "Incorrect accuracy value selected for n=200"

    def test_aggregate_mixed_metrics(self, tmpdir: Path, write_data_to_file: Callable) -> None:
        """
        Test the aggregate function with mixed metrics (R² and accuracy).

        This test verifies that:
        1. The output contains the R² column
        2. The output does not contain the accuracy column
        3. The number of rows in the output is correct

        Args:
            tmpdir: Pytest fixture for temporary directory.
            write_data_to_file: Fixture to write data to a file.
        """
        scores1 = {"n": [100, 200], "s": [42, 43], "r2_val": [0.8, 0.9]}
        scores2 = {"n": [100, 200], "s": [42, 43], "acc_val": [0.75, 0.85]}

        scores_path1 = write_data_to_file(pd.DataFrame(scores1), 'csv', tmpdir / "scores1.csv")
        scores_path2 = write_data_to_file(pd.DataFrame(scores2), 'csv', tmpdir / "scores2.csv")

        score_path_list = [str(scores_path1), str(scores_path2)]
        stats_path = str(tmpdir / "stats.csv")

        aggregate(score_path_list, stats_path)

        stats = pd.read_csv(stats_path)

        assert "r2_val" in stats.columns, "r2_val column is missing in the output"
        assert "acc_val" not in stats.columns, "acc_val column should not be present in the output"
        assert len(stats) == 2, "Incorrect number of rows in the output"

    def test_aggregate_duplicate_max_scores(self, tmpdir: Path, write_data_to_file: Callable) -> None:
        """
        Test the aggregate function with duplicate maximum scores.

        This test verifies that:
        1. Only one row is selected when there are duplicate maximum scores
        2. The correct row is selected (first occurrence)

        Args:
            tmpdir: Pytest fixture for temporary directory.
            write_data_to_file: Fixture to write data to a file.
        """
        scores = {"n": [100, 100], "s": [42, 42], "r2_val": [0.8, 0.8], "other_param": ["a", "b"]}

        scores_path = write_data_to_file(pd.DataFrame(scores), 'csv', tmpdir / "scores.csv")

        score_path_list = [str(scores_path)]
        stats_path = str(tmpdir / "stats.csv")

        aggregate(score_path_list, stats_path)

        stats = pd.read_csv(stats_path)

        assert len(stats) == 1, "Incorrect number of rows in the output"
        assert stats["other_param"].values[0] == "a", "Incorrect row selected for duplicate max scores"

    def test_aggregate_varying_input_files(self, tmpdir: Path, write_data_to_file: Callable) -> None:
        """
        Test the aggregate function with varying numbers of input files.

        This test verifies that:
        1. The correct number of rows is present in the output
        2. The correct sample sizes are present in the output

        Args:
            tmpdir: Pytest fixture for temporary directory.
            write_data_to_file: Fixture to write data to a file.
        """
        scores1 = {"n": [100], "s": [42], "r2_val": [0.8]}
        scores2 = {"n": [200], "s": [43], "r2_val": [0.9]}
        scores3 = {"n": [300], "s": [44], "r2_val": [0.85]}

        scores_path1 = write_data_to_file(pd.DataFrame(scores1), 'csv', tmpdir / "scores1.csv")
        scores_path2 = write_data_to_file(pd.DataFrame(scores2), 'csv', tmpdir / "scores2.csv")
        scores_path3 = write_data_to_file(pd.DataFrame(scores3), 'csv', tmpdir / "scores3.csv")

        score_path_list = [str(scores_path1), str(scores_path2), str(scores_path3)]
        stats_path = str(tmpdir / "stats.csv")

        aggregate(score_path_list, stats_path)

        stats = pd.read_csv(stats_path)

        assert len(stats) == 3, "Incorrect number of rows in the output"
        assert set(stats["n"]) == {100, 200, 300}, "Incorrect sample sizes in the output"

    def test_aggregate_reproducibility(self, tmpdir: Path, write_data_to_file: Callable) -> None:
        """
        Test the reproducibility of the aggregate function.

        This test verifies that:
        1. Running the aggregate function twice with the same input produces identical results

        Args:
            tmpdir: Pytest fixture for temporary directory.
            write_data_to_file: Fixture to write data to a file.
        """
        scores1 = {"n": [100, 200], "s": [42, 43], "r2_val": [0.8, 0.9], "other_param": ["a", "b"]}
        scores2 = {"n": [100, 200], "s": [42, 43], "r2_val": [0.7, 0.85], "other_param": ["c", "d"]}

        scores_path1 = write_data_to_file(pd.DataFrame(scores1), 'csv', tmpdir / "scores1.csv")
        scores_path2 = write_data_to_file(pd.DataFrame(scores2), 'csv', tmpdir / "scores2.csv")

        score_path_list = [str(scores_path1), str(scores_path2)]
        stats_path1 = str(tmpdir / "stats1.csv")
        stats_path2 = str(tmpdir / "stats2.csv")

        aggregate(score_path_list, stats_path1)
        aggregate(score_path_list, stats_path2)

        stats1 = pd.read_csv(stats_path1)
        stats2 = pd.read_csv(stats_path2)

        pd.testing.assert_frame_equal(stats1, stats2, "Results are not reproducible")