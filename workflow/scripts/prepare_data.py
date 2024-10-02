from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml


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
    # Check if the dataset is predefined and the variant exists
    if (
        dataset in predefined_datasets
        and variant in predefined_datasets[dataset][features_targets_covariates]
    ):
        # Load data using the predefined lambda function
        data = predefined_datasets[dataset][features_targets_covariates][variant]()
    elif features_targets_covariates == "covariates" and variant == 'none':
        # Handle cases where no covariates are needed
        data = np.array([])
    else:
        # Load custom datasets based on file extension
        in_path = Path(custom_datasets[dataset][features_targets_covariates][variant])
        if in_path.suffix == ".csv":
            data = pd.read_csv(in_path).values
        elif in_path.suffix == ".tsv":
            data = pd.read_csv(in_path, delimiter="\t").values
        elif in_path.suffix == ".npy":
            data = np.load(in_path)
        else:
            raise ValueError(f"Unsupported file format: {in_path.suffix}")
    
    # Apply appropriate masking based on the type of data
    if features_targets_covariates == "targets":
        data = data.reshape(-1)
        mask = np.isfinite(data)
    elif features_targets_covariates == "features":
        assert np.ndim(data) == 2, "Features data must be two-dimensional."
        mask = np.isfinite(data).all(axis=1)
    elif features_targets_covariates == "covariates" and data.size > 0:
        if np.ndim(data) == 1:
            data = data.reshape(-1, 1)
        mask = np.isfinite(data).all(axis=1)
    else:
        mask = np.array([], dtype=bool)
    
    # Save the processed data and mask to an HDF5 file
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("mask", data=mask)


if __name__ == "__main__":
    """
    Entry point for the script when executed as a standalone program.
    Parses parameters from Snakemake and initiates the data preparation process.
    """
    prepare_data(
        out_path=snakemake.output.out,
        dataset=snakemake.wildcards.dataset,
        features_targets_covariates=snakemake.wildcards.features_or_targets
        if hasattr(snakemake.wildcards, "features_or_targets")
        else "covariates",
        variant=snakemake.wildcards.name,
        custom_datasets=snakemake.params.custom_datasets,
    )