import logging
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Set up logging
log_level = os.environ.get('ESCE_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

# Define a constant for floating point precision
FLOAT_PRECISION = np.float32

predefined_datasets = {
    "mnist": {
        "features": {
            "pixel": lambda: fetch_openml(
                "mnist_784", version=1, return_X_y=True, as_frame=False
            )[0]
        },
        "targets": {
            "ten-digits": lambda: fetch_openml(
                "mnist_784", version=1, return_X_y=True, as_frame=False
            )[1].astype(int),
            "odd-even": lambda: (
                fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)[
                    1
                ].astype(int)
                % 2
            ).astype(int),
        },
        "covariates": {},
    }
}

def prepare_data(
    out_path: str,
    dataset: str,
    features_targets_covariates: str,
    variant: str,
    custom_datasets: dict,
):
    """
    Prepare a dataset for use in the workflow by loading, processing, and saving it in HDF5 format.

    This function handles predefined datasets and custom datasets. It reads data from CSV, TSV, or NPY files,
    performs necessary preprocessing, applies masks to filter valid samples, and saves the processed data
    along with the mask into an HDF5 file.

    Args:
        out_path (str): Path to save the resulting HDF5 file.
        dataset (str): Name of the dataset to prepare.
        features_targets_covariates (str): Type of data to load ('features', 'targets', or 'covariates').
        variant (str): Specific variant of the dataset to load.
        custom_datasets (dict): Dictionary containing paths to custom datasets.
    """
    logging.info(f"Preparing data for dataset: {dataset}, type: {features_targets_covariates}, variant: {variant}")

    # Check if the dataset is predefined and the variant exists
    if dataset in predefined_datasets and variant in predefined_datasets[dataset].get(features_targets_covariates, {}):
        logging.debug(f"Loading predefined dataset: {dataset}")
        # Load data using the predefined lambda function
        data = predefined_datasets[dataset][features_targets_covariates][variant]()
    elif features_targets_covariates == "covariates" and variant == 'none':
        logging.debug("No covariates specified, creating empty array")
        # Handle cases where no covariates are needed
        data = np.array([])
    else:
        # Load custom datasets based on file extension
        if dataset not in custom_datasets or features_targets_covariates not in custom_datasets[dataset] or variant not in custom_datasets[dataset][features_targets_covariates]:
            error_msg = "Requested predefined dataset or variant does not exist."
            logging.error(error_msg)
            raise KeyError(error_msg)
        
        in_path = Path(custom_datasets[dataset][features_targets_covariates][variant])
        logging.info(f"Loading custom dataset from: {in_path}")
        if in_path.suffix == ".csv":
            data = pd.read_csv(in_path).values
        elif in_path.suffix == ".tsv":
            data = pd.read_csv(in_path, delimiter="\t").values
        elif in_path.suffix == ".npy":
            data = np.load(in_path)
        else:
            error_msg = "Unsupported file format. Supported formats are CSV, TSV, and NPY."
            logging.error(error_msg)
            raise ValueError(error_msg)

    # Check if the dataset is empty after loading
    if data.size == 0 and features_targets_covariates != "covariates":
        error_msg = "Dataset is empty after loading."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Check for incompatible data types
    if not np.issubdtype(data.dtype, np.number):
        error_msg = f"Incompatible data type: {data.dtype}. Numeric data is required."
        logging.error(error_msg)
        raise TypeError(error_msg)

    # Apply appropriate processing and masking based on the type of data
    if features_targets_covariates == "targets":
        data = data.reshape(-1)
        if data.ndim != 1:
            error_msg = "Targets data must be 1D after reshaping."
            logging.error(error_msg)
            raise AssertionError(error_msg)
        mask = np.isfinite(data)
    elif features_targets_covariates == "features":
        if data.ndim != 2:
            error_msg = "Features data must be 2D."
            logging.error(error_msg)
            raise AssertionError(error_msg)
        mask = np.isfinite(data).all(axis=1)
    elif features_targets_covariates == "covariates":
        if data.size > 0:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            mask = np.isfinite(data).all(axis=1)
        else:
            mask = np.array([], dtype=bool)
    else:
        error_msg = f"Invalid data type: {features_targets_covariates}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.info(f"Data shape: {data.shape}, Mask shape: {mask.shape}")
    logging.debug(f"Number of valid samples: {np.sum(mask)}")

    # Save the processed data and mask to an HDF5 file
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data.astype(FLOAT_PRECISION))
        f.create_dataset("mask", data=mask)

    logging.info(f"Data saved to HDF5 file: {out_path}")

if __name__ == "__main__":
    """
    Entry point for the script when executed as a standalone program.
    Parses parameters from Snakemake and initiates the data preparation process.
    """
    logging.info("Starting data preparation process")
    prepare_data(
        out_path=snakemake.output.out,
        dataset=snakemake.wildcards.dataset,
        features_targets_covariates=snakemake.wildcards.features_or_targets
        if hasattr(snakemake.wildcards, "features_or_targets")
        else "covariates",
        variant=snakemake.wildcards.name,
        custom_datasets=snakemake.params.custom_datasets,
    )
    logging.info("Data preparation process completed")
