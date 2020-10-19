import shelve
import numpy
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.svm import SVC
from joblib import hash
import numbers
import pandas
import numpy as np
import math

from esce.grid import GRID
from esce.util import cached

@cached("cache/gram.h5")
def get_gram_tril(data, gamma=None):
    """Calculates the lower triangle of the gram matrix.

    Args:
        data: Data to compute gram matrix of.
        gamma: RBF kernel gamma.

    Returns:
        One-dimensional array containing the lower triangle
        of the computed gram matrix.
    """
    x = data.astype(np.float32)
    if gamma is None:
        K = linear_kernel(x, x)
    elif isinstance(gamma, numbers.Number):
        K = rbf_kernel(x, x, gamma=gamma)
    else:
        raise ValueError
    return K[np.tril_indices(K.shape[0])]

def get_gram(data, gamma=None):
    """Reconstructs the gram matrix based on lower triangle.

    Args:
        data: Data to compute gram matrix of.
        gamma: RBF kernel gamma.

    Returns:
        Two-dimensional gram matrix of the data.
    """
    tri = get_gram_tril(data, gamma)
    n = int(0.5 * (math.sqrt(8 * len(tri) + 1) - 1))
    K = np.zeros((n,n))
    K[np.tril_indices(n)] = tri

    # TODO: make this more efficient memory-wise?
    K = K + K.T - np.diag(np.diag(K))
    return K

def score(gram, y, C, idx_train, idx_val, idx_test):
    model = SVC(C=C, kernel='precomputed', max_iter=1000)

    gram_ = gram[numpy.ix_(idx_train, idx_train)]
    model.fit(gram_, y[idx_train])

    gram_ = gram[numpy.ix_(idx_val, idx_train)]
    score_val = model.score(gram_, y[idx_val])

    gram_ = gram[numpy.ix_(idx_test, idx_train)]
    score_test = model.score(gram_, y[idx_test])

    return score_val, score_test


def score_splits(x, y, splits):
    results = []

    for model in ['linear', 'rbf']:
        for gamma in GRID[model]['gamma']:
            gram = get_gram(x, gamma=gamma)

            for n in splits:
                for s in splits[n]:
                    idx_train, idx_val, idx_test = splits[n][s]

                    for C in GRID[model]['C']:
                        score_val, score_test = score(gram, y, C, idx_train, idx_val, idx_test)
                        results.append([n, s, model, C, gamma, score_val, score_test])
                        print(results[-1])

    results = pandas.DataFrame(results, columns=['n', 's', 'model', 'C', 'gamma', 'score_val', 'score_test'])
    return results
