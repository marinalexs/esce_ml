from torchvision.datasets import MNIST, FashionMNIST
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from zipfile import ZipFile
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import hash

from esce.util import download_file

def get_mnist():
    ds = MNIST('data/', train=True, download=True)
    x, y = ds.data.numpy(), ds.targets.numpy()
    x = x.reshape(len(x), -1)
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)

    y2 = y % 2 == 0
    return x, {"default": y, "binary": y2}

def get_fashion_mnist():
    ds = FashionMNIST("data/", train=True, download=True)
    x, y = ds.data.numpy(), ds.targets.numpy()
    x = x.reshape(len(x), -1)
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    return x,y

def get_superconductivity():
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