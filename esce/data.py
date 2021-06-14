"""This module provides datasets for ESCE.

The following datasets are available: MNIST, FashionMNIST, superconduct, HIGGS.
"""

import codecs
from pathlib import Path
from typing import Tuple, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from esce.util import download_file, extract_gzip

SN3_TYPEMAP = {
    8: np.uint8,
    9: np.int8,
    11: np.dtype(">i2"),
    12: np.dtype(">i4"),
    13: np.dtype(">f4"),
    14: np.dtype(">f8"),
}


def get_int(b: bytes) -> int:
    """Convert bytes to integer.

    Argument:
        b: bytes to convert
    """
    return int(codecs.encode(b, "hex"), 16)


def read_sn3_tensor(path: Path) -> np.ndarray:
    """Read SN3/ubyte file format.

    Argument:
        path: Path to dataset

    Returns:
        numpy array containing the data
    """
    with path.open("rb") as f:
        data = f.read()
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = SN3_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m, offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s)
    return cast(np.ndarray, parsed.reshape(*s))


def get_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the MNIST dataset.

    Returns:
        Tuple (x,y) containing the training data and the labels.
    """
    train_images_path = Path("data/MNIST/raw/train-images-idx3-ubyte")
    train_labels_path = Path("data/MNIST/raw/train-labels-idx1-ubyte")
    train_images_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_images_path.is_file():
        train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        gzip_path = train_images_path.with_suffix(".gz")
        download_file(train_images_url, gzip_path)
        extract_gzip(gzip_path, train_images_path)

    if not train_labels_path.is_file():
        train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        gzip_path = train_labels_path.with_suffix(".gz")
        download_file(train_labels_url, gzip_path)
        extract_gzip(gzip_path, train_labels_path)

    x = read_sn3_tensor(train_images_path)
    y = read_sn3_tensor(train_labels_path)
    x = x.reshape(len(x), -1)
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    return x, y


def get_fashion_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the FashionMNIST dataset.

    Returns:
        Tuple (x,y) containing the training data and the labels.
    """
    train_images_path = Path("data/FashionMNIST/raw/train-images-idx3-ubyte")
    train_labels_path = Path("data/FashionMNIST/raw/train-labels-idx1-ubyte")
    train_images_path.parent.mkdir(parents=True, exist_ok=True)

    if not train_images_path.is_file():
        train_images_url = (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
            "train-images-idx3-ubyte.gz"
        )
        gzip_path = train_images_path.with_suffix(".gz")
        download_file(train_images_url, gzip_path)
        extract_gzip(gzip_path, train_images_path)

    if not train_labels_path.is_file():
        train_labels_url = (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
            "train-labels-idx1-ubyte.gz"
        )
        gzip_path = train_labels_path.with_suffix(".gz")
        download_file(train_labels_url, gzip_path)
        extract_gzip(gzip_path, train_labels_path)

    x = read_sn3_tensor(train_images_path)
    y = read_sn3_tensor(train_labels_path)
    x = x.reshape(len(x), -1)
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    return x, y


def get_superconductivity() -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the superconductivity/superconduct dataset.

    Returns:
        Tuple (x,y) containing the training data and the labels.
    """
    csv_path = Path("data/superconductivity.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = Path("data/superconduct.zip")

    if not csv_path.is_file():
        if not zip_path.is_file():
            print("Downloading superconduct.zip...")
            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "00464/superconduct.zip"
            )
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
    """Retrieve the HIGGS dataset.

    Returns:
        Tuple (x,y) containing the training data and the labels.
    """
    gz_path = Path("data/HIGGS.csv.gz")
    gz_path.parent.mkdir(parents=True, exist_ok=True)

    if not gz_path.is_file():
        print("Downloading HIGGS.csv.gz...")
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00280/HIGGS.csv.gz"
        )
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
