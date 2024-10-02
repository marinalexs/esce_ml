"""
aggregate.py
====================================

Aggregates the scores for each hyperparameter combination and selects the best
hyperparameter combination, grouped by sample size and seed.

This script reads multiple score files, identifies the best-performing hyperparameter
combination based on a single validation metric (either R² or accuracy), and consolidates 
the results into a single statistics CSV file.

"""

import os
from pathlib import Path
from typing import List

import pandas as pd


def aggregate(
    score_path_list: List[str],
    stats_path: str,
) -> None:
    """
    Aggregate scores from multiple files and identify the best hyperparameter combinations.

    For each score file in `score_path_list`, the function identifies the hyperparameter
    combination with the highest validation metric (either R² or accuracy) for each group 
    defined by sample size (`n`) and seed (`s`). It then compiles these best scores into 
    a single CSV file at `stats_path`.

    Args:
        score_path_list (List[str]): List of file paths to the input score CSV files.
        stats_path (str): Path to save the aggregated statistics CSV file.

    Returns:
        None: The function saves the results to a CSV file but doesn't return any value.
    """
    df_list = []
    for filename in score_path_list:
        # Ignore empty files (indicating insufficient samples in the dataset)
        if os.stat(filename).st_size > 0:
            df_list.append(pd.read_csv(filename, index_col=False))

    # If no valid score files are found, create an empty output file and exit
    if not df_list:
        Path(stats_path).touch()
        return

    # Concatenate all score DataFrames into a single DataFrame
    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Determine the validation metric to use (R² for regression, accuracy for classification)
    if "r2_val" in df.columns:
        metric = "r2_val"
    elif "acc_val" in df.columns:
        metric = "acc_val"
    else:
        raise ValueError("No valid metric (r2_val or acc_val) found in the input files.")

    # Group the DataFrame by sample size ('n') and seed ('s')
    # For each group, identify the index of the row with the maximum validation metric
    idx_best = df.groupby(["n", "s"])[metric].idxmax()

    # Select the best-performing hyperparameter combinations based on the identified indices
    df_best = df.loc[idx_best]

    # Save the aggregated best scores to the specified statistics CSV file
    df_best.to_csv(stats_path, index=False)


if __name__ == "__main__":
    """
    Entry point for the script when executed as a standalone program.
    Parses parameters from Snakemake and initiates the aggregation process.
    """
    aggregate(
        score_path_list=snakemake.input.scores,
        stats_path=snakemake.output.scores
    )