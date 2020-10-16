from torchvision.datasets import MNIST, FashionMNIST
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import numpy
from joblib import hash

def get_mnist(n_components=8, method='pca', noise=None):
    ds = MNIST('data/', train=True, download=True)
    x, y = ds.data.numpy(), ds.targets.numpy()
    x = x.reshape(len(x), -1)

    x, _, y, _ = train_test_split(x, y, train_size=12000, random_state=0)
    x = StandardScaler().fit_transform(x)
    if method == 'pca':
        x = PCA(n_components=n_components, whiten=True, random_state=0).fit_transform(x)
    elif method == 'rp':
        x = GaussianRandomProjection(n_components=n_components, random_state=0).fit_transform(x)
    else:
        raise ValueError
    x = StandardScaler().fit_transform(x)

    if noise is not None:
        x = (x + noise * numpy.random.randn(*x.shape)) / numpy.sqrt(1 + noise ** 2)

    return x, y
