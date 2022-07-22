from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import StandardScaler


def download_mnist(
    x_path: str,
    y_path: str,
) -> None:

    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    x = StandardScaler().fit_transform(x)
    np.save(x_path, x.astype(int))
    np.save(y_path, y.astype(int))


download_mnist(snakemake.output.x, snakemake.output.y)
