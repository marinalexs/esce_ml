"""
aggregate.py
====================================

Aggregates the scores for each hyperparameter combination and selects the best
hyperparameter combination, grouped by sample size and seed.

This script reads multiple score files, identifies the best-performing hyperparameter
combination based on validation metrics (R² or accuracy), and consolidates the results
into a single statistics CSV file.

"""

import os
from pathlib import Path
from typing import List
import logging

import pandas as pd

# Set up logging
log_level = os.environ.get('ESCE_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate(
    score_path_list: List[str],
    stats_path: str,
) -> None:
    """
    Aggregate scores from multiple files and identify the best hyperparameter combinations.

    For each score file in `score_path_list`, the function identifies the hyperparameter
    combination with the highest validation metric (R² or accuracy) for each group defined
    by sample size (`n`) and seed (`s`). It then compiles these best scores into a single
    CSV file at `stats_path`.

    Args:
        score_path_list (List[str]): List of file paths to the input score CSV files.
        stats_path (str): Path to save the aggregated statistics CSV file.

    Returns:
        None: The function saves the results to a CSV file but doesn't return any value.
    """
    logging.info(f"Starting aggregation process with {len(score_path_list)} input files.")
    df_list = []
    for filename in score_path_list:
        # Ignore empty files (indicating insufficient samples in the dataset)
        if os.stat(filename).st_size > 0:
            try:
                df = pd.read_csv(filename, index_col=False)
                if not df.empty:
                    df_list.append(df)
                    logging.debug(f"Successfully read file: {filename}")
                else:
                    logging.warning(f"File {filename} is empty.")
            except pd.errors.EmptyDataError:
                logging.warning(f"File {filename} is empty.")
            except Exception as e:
                logging.error(f"Error reading file {filename}: {str(e)}")
        else:
            logging.warning(f"Skipping empty file: {filename}")

    # If no valid score files are found, create an empty output file and exit
    if not df_list:
        Path(stats_path).touch()
        logging.warning("No valid score files found. Created an empty output file.")
        return

    # Concatenate all score DataFrames into a single DataFrame
    df = pd.concat(df_list, axis=0, ignore_index=True)
    logging.info(f"Concatenated {len(df_list)} DataFrames with a total of {len(df)} rows.")

    # Determine the validation metric to use (R² for regression, accuracy for classification)
    # Prioritize R² if present
    metric = "r2_val" if "r2_val" in df.columns else "acc_val" if "acc_val" in df.columns else None
    if metric is None:
        logging.error("No valid metric column (r2_val or acc_val) found in the data.")
        return

    logging.info(f"Using {metric} as the validation metric.")

    # Check if both 'n' and 's' columns are present
    if 'n' not in df.columns or 's' not in df.columns:
        logging.error("Both 'n' and 's' columns must be present in the data.")
        return

    # Group the DataFrame by sample size ('n') and seed ('s')
    # For each group, identify the index of the row with the maximum validation metric
    idx_best = df.groupby(["n", "s"])[metric].idxmax()
    logging.debug(f"Identified {len(idx_best)} best-performing hyperparameter combinations.")

    # Select the best-performing hyperparameter combinations based on the identified indices
    df_best = df.loc[idx_best]

    # Keep only the columns that are present in all rows
    columns_to_keep = df_best.columns[df_best.notna().all()].tolist()
    df_best = df_best[columns_to_keep]

    # Save the aggregated best scores to the specified statistics CSV file
    df_best.to_csv(stats_path, index=False)
    logging.info(f"Aggregated results saved to {stats_path}")

if __name__ == "__main__":
    """
    Entry point for the script when executed as a standalone program.
    Parses parameters from Snakemake and initiates the aggregation process.
    """
    logging.info("Starting aggregate.py script")
    aggregate(
        score_path_list=snakemake.input.scores,
        stats_path=snakemake.output.scores
    )
    logging.info("Finished aggregate.py script")