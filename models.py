import shelve

import numpy
from grid import GRID
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.svm import SVC
from joblib import hash
import h5py
import numbers
import pandas


# def get_gram(x, gamma=None):
#     x_hash = hash(x)
#     key = f'{x_hash}/{gamma}'
#
#     with h5py.File('tmp.h5', 'a') as f:
#         if key not in f:
#             f.create_dataset(key, (len(x), len(x)), dtype='f')
#             if gamma is None:
#                 f[key][...] = linear_kernel(x, x)
#             elif isinstance(gamma, numbers.Number):
#                 f[key][...] = rbf_kernel(x, x, gamma=gamma)
#             else:
#                 raise ValueError
#             print('calc', x_hash, gamma)
#         else:
#             print('cache', x_hash, gamma)
#
#         return f[key][...]

def get_gram(x, gamma=None):
    if gamma is None:
        return linear_kernel(x, x)
    elif isinstance(gamma, numbers.Number):
        return rbf_kernel(x, x, gamma=gamma)
    else:
        raise ValueError


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
