import shelve
import numpy
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.svm import SVC, SVR
from joblib import hash
import numbers
import pandas
import numpy as np
import math
from enum import Enum
from sklearn.model_selection import ParameterGrid
from abc import ABC, abstractmethod

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

def score_splits(x, y, splits):
    results = []

    for model_name in MODELS:
        model = MODELS[model_name]

        for n in splits:
            for s in splits[n]:
                for params in ParameterGrid(GRID[model_name]):
                    idx_train, idx_val, idx_test = splits[n][s]
                    score_val, score_test = model.score(x, y, idx_train, idx_val, idx_test, **params)

                    # TODO: change this
                    gamma = params["gamma"] if "gamma" in params else 1
                    results.append([n,s,model_name, params['C'], gamma, score_val, score_test])
                    print(results[-1])

    results = pandas.DataFrame(results, columns=['n', 's', 'model', 'C', 'gamma', 'score_val', 'score_test'])
    return results

MODELS = {
    "svm-linear": KernelSVMModel(kernel=KernelType.LINEAR),
    "svm-rbf": KernelSVMModel(kernel=KernelType.RBF)
}