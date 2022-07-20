from sklearn.datasets import fetch_openml
import numpy as np

def download_mnist(
        x_path: str,
        y_path: str,) -> None:

    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    print(x,y)
    np.savetxt(x_path, x, delimiter=',')
    np.savetxt(y_path, y.astype(int).reshape(-1,1), delimiter=',')

download_mnist(snakemake.output.x, snakemake.output.y)
