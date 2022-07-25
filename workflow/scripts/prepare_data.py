from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd


def download_mnist(
    out_path: str,
    dataset: str,
    features_targets_covariates: str,
    variant: str,
    custom_datasets: dict,
) -> None:
    if dataset == 'mnist' and variant == 'pixel':
        x, _ = fetch_openml("mnist_784", version=1,
                            return_X_y=True, as_frame=False)
        data = x
    elif dataset == 'mnist' and variant == 'ten-digits':
        _, y = fetch_openml("mnist_784", version=1,
                            return_X_y=True, as_frame=False)
        data = y.astype(int)
    elif dataset == 'mnist' and variant == 'odd-even':
        _, y = fetch_openml("mnist_784", version=1,
                            return_X_y=True, as_frame=False)
        data = (y.astype(int) % 2 == 0).astype(int)
    elif features_targets_covariates == 'covariates' and variant in ['none', 'balanced']:
        data = []
    else:
        in_path = Path(custom_datasets[dataset][features_targets_covariates][variant])
        if in_path.suffix == '.csv':
            data = pd.read_csv(in_path).values
        if in_path.suffix == '.tsv':
            data = pd.read_csv(in_path, delimiter='\t').values
        if in_path.suffix == '.npy':
            data = np.load(in_path)
    if features_targets_covariates == 'features':
        data = StandardScaler().fit_transform(data)
    if features_targets_covariates == 'targets':
        data = data.reshape(-1)
    np.save(out_path, data)


download_mnist(
    snakemake.output.npy,
    snakemake.wildcards.dataset,
    snakemake.wildcards.features_or_targets if hasattr(snakemake.wildcards, 'features_or_targets') else 'covariates',
    snakemake.wildcards.name,
    snakemake.params.custom_datasets,
)
