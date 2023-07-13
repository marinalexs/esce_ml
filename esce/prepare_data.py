from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from esce.predefined_datasets import predefined_datasets


def prepare_data(
    out_path: str,
    dataset: str,
    features_targets_covariates: str,
    variant: str,
    custom_datasets: dict,
):
    print(dataset, features_targets_covariates, variant)
    if (
        dataset in predefined_datasets
        and variant in predefined_datasets[dataset][features_targets_covariates]
    ):
        data = predefined_datasets[dataset][features_targets_covariates][variant]()
    elif features_targets_covariates == "covariates" and variant in [
        "none",
        "balanced",
    ]:
        data = np.array([])
    else:
        in_path = Path(custom_datasets[dataset][features_targets_covariates][variant])
        if in_path.suffix == ".csv":
            data = pd.read_csv(in_path).values
        if in_path.suffix == ".tsv":
            data = pd.read_csv(in_path, delimiter="\t").values
        if in_path.suffix == ".npy":
            data = np.load(in_path)

    if features_targets_covariates == "targets":
        data = data.reshape(-1)
        mask = np.isfinite(data)
    elif features_targets_covariates == "features":
        assert np.ndim(data) == 2
        mask = np.isfinite(data).all(axis=1)
    elif features_targets_covariates == "covariates" and len(data) > 0:
        if np.ndim(data) == 1:
            data = data.reshape(-1, 1)
        mask = np.isfinite(data).all(axis=1)
    else:
        mask = np.array([])

    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("mask", data=mask)
