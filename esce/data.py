from pathlib import Path
from typing import Dict, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from esce.util import download_file


# todo: remove torchvision dependency
def get_mnist() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    from torchvision.datasets import MNIST

    ds = MNIST("data/", train=True, download=True)
    x, y = ds.data.numpy(), ds.targets.numpy()
    x = x.reshape(len(x), -1)
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    return x, y


def get_fashion_mnist() -> Tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import FashionMNIST

    ds = FashionMNIST("data/", train=True, download=True)
    x, y = ds.data.numpy(), ds.targets.numpy()
    x = x.reshape(len(x), -1)
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    return x, y


def get_superconductivity() -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path("data/superconductivity.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = Path("data/superconduct.zip")

    if not csv_path.is_file():
        if not zip_path.is_file():
            print("Downloading superconduct.zip...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
            download_file(url, zip_path)
        with ZipFile(zip_path, "r") as zipfile:
            with zipfile.open("train.csv") as src:
                with csv_path.open("wb") as dst:
                    dst.write(src.read())

    df = pd.read_csv(csv_path)
    x = df.values[:, :-1]
    y = df.values[:, -1]
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    return x, y


def get_higgs() -> Tuple[np.ndarray, np.ndarray]:
    gz_path = Path("data/HIGGS.csv.gz")

    if not gz_path.is_file():
        print("Downloading HIGGS.csv.gz...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"  # noqa
        download_file(url, gz_path)

    df = pd.read_csv(gz_path, nrows=12000)
    x = df.values[:, 1:]
    y = df.values[:, 0]
    x = StandardScaler().fit_transform(x)
    return x, y


DATA = {
    "mnist": get_mnist,
    "fashion": get_fashion_mnist,
    "superconductivity": get_superconductivity,
    "higgs": get_higgs,
}
