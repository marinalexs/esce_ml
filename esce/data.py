from torchvision.datasets import MNIST, FashionMNIST
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from joblib import hash

def get_mnist():
    ds = MNIST('data/', train=True, download=True)
    x, y = ds.data.numpy(), ds.targets.numpy()
    x = x.reshape(len(x), -1)
    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    return x, y

def get_fashion_mnist():
    pass