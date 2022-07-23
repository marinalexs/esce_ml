from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def download_mnist(
    x_raw_path: str,
    y_raw_path: str,
    y_odd_path: str,
) -> None:

    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    x = StandardScaler().fit_transform(x)
    np.save(x_raw_path, x.astype(float))
    np.save(y_raw_path, y.astype(int))
    np.save(y_odd_path, (y.astype(int) % 2 == 0).astype(int))


download_mnist(
    snakemake.output.x_raw,
    snakemake.output.x_pca,
    snakemake.output.y_raw,
    snakemake.output.y_odd,
)
