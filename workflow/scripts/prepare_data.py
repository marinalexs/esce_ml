from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def prepare_data(
    out_path: str,
    dataset: str,
    features_targets_covariates: str,
    variant: str,
    custom_datasets: dict,
):
    if (dataset, variant) in predefined_datasets:
        data = predefined_datasets[(dataset, variant)]()
    elif features_targets_covariates == "covariates" and variant in [
        "none",
        "balanced",
    ]:
        data = []
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

    np.save(out_path, data)

if __name__ == "__main__":
    prepare_data(
        snakemake.output.npy,
        snakemake.wildcards.dataset,
        snakemake.wildcards.features_or_targets
        if hasattr(snakemake.wildcards, "features_or_targets")
        else "covariates",
        snakemake.wildcards.name,
        snakemake.params.custom_datasets,
    )
