import shelve
import numpy
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from joblib import hash
import numbers
import pandas as pd
import numpy as np
import math
from enum import Enum
from sklearn.model_selection import ParameterGrid
from abc import ABC, abstractmethod
import csv
from pathlib import Path

from esce.grid import GRID
from esce.util import cached

class KernelType(Enum):
    LINEAR = 1
    RBF = 2

@cached("cache/gram.h5")
def get_gram_triu(data, kernel=KernelType.LINEAR, gamma=0):
    """Calculates the upper triangle of the gram matrix.

    Args:
        data: Data to compute gram matrix of.
        kernel: Kernel type
        gamma: RBF kernel gamma.

    Returns:
        One-dimensional array containing the upper triangle
        of the computed gram matrix.
    """
    x = data.astype(np.float32)
    if kernel == KernelType.LINEAR:
        K = linear_kernel(x, x)
    elif kernel == KernelType.RBF:
        K = rbf_kernel(x, x, gamma=gamma)
    else:
        raise ValueError
    return K[np.triu_indices(K.shape[0])]

def get_gram(data, kernel=KernelType.LINEAR, gamma=0):
    """Reconstructs the gram matrix based on upper triangle.

    Args:
        data: Data to compute gram matrix of.
        gamma: RBF kernel gamma.

    Returns:
        Two-dimensional gram matrix of the data.
    """
    tri = get_gram_triu(data, kernel, gamma)
    n = int(0.5 * (math.sqrt(8 * len(tri) + 1) - 1))
    K = np.zeros((n,n), dtype=np.float32)
    K[np.triu_indices(n)] = tri

    # TODO: make this more efficient memory-wise?
    K = K + K.T - np.diag(np.diag(K))
    return K

class BaseModel(ABC):
    @abstractmethod
    def score(self, x, y, idx_train, idx_val, idx_test, params):
        pass

class OLSModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def score(self, x, y, idx_train, idx_val, idx_test): 
        self.model.fit(x[idx_train], y[idx_train])

        score_val = self.model.score(x[idx_val], y[idx_val])
        score_test = self.model.score(x[idx_test], y[idx_test])
        return score_val, score_test

class KernelSVMModel(BaseModel):
    def __init__(self, kernel=KernelType.LINEAR):
        self.kernel = kernel
        self.prev_gamma = None
        self.cached_gram = None

    def get_gram(self, x, gamma):
        if self.prev_gamma == gamma:
            return self.cached_gram
        else:
            self.prev_gamma = gamma
            self.cached_gram = get_gram(x, kernel=self.kernel, gamma=gamma)
            return self.cached_gram

    def score(self, x, y, idx_train, idx_val, idx_test, C=1, gamma=0):
        gram = self.get_gram(x, gamma)
        model = SVC(C=C, kernel='precomputed', max_iter=1000)

        gram_ = gram[np.ix_(idx_train, idx_train)]
        model.fit(gram_, y[idx_train])

        gram_ = gram[np.ix_(idx_val, idx_train)]
        score_val = model.score(gram_, y[idx_val])

        gram_ = gram[np.ix_(idx_test, idx_train)]
        score_test = model.score(gram_, y[idx_test])

        return score_val, score_test

def score_splits(outfile, x, y, splits, warm_start=False):
    prev_run_file = Path(outfile)
    if prev_run_file.is_file() and warm_start:
        df = pd.read_csv(outfile)
    else:
        with open(outfile, "w") as f:
            f.write("model,n,param_hash,score_val,score_test\n")
        df = pd.read_csv(outfile)
    
    with open(outfile, "a") as f:
        csvwriter = csv.writer(f, delimiter=",")

        for model_name in MODELS:
            model = MODELS[model_name]

            for n in splits:
                for params in ParameterGrid(GRID[model_name]):
                    param_hash = hash(params)
                    if not ((df["model"] == model_name) & (df["n"] == n) & (df["param_hash"] == param_hash)).any():
                        idx_train, idx_val, idx_test = splits[n]
                        score_val, score_test = model.score(x, y, idx_train, idx_val, idx_test, **params)
                        print(model_name, n, param_hash, score_val, score_test)

                        csvwriter.writerow([model_name, n, param_hash, score_val, score_test])
                        f.flush()

MODELS = {
    "ols": OLSModel(),
    "svm-linear": KernelSVMModel(kernel=KernelType.LINEAR),
    "svm-rbf": KernelSVMModel(kernel=KernelType.RBF)
}