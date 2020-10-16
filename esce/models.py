import shelve
import numpy
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.svm import SVC
from joblib import hash
import numbers
import pandas

from esce.grid import GRID
from esce.util import cached

# def get_gram(x, gamma=None):
#     x_hash = hash(x)
#     key = f'{x_hash}/{gamma}'
#     path = Path("cache/gram.h5")
#     path.parent.mkdir(parents=True, exist_ok=True)

#     with h5py.File(path, 'a') as f:
#         if key not in f:
#             dset = f.create_dataset(key, (len(x), len(x)), dtype='f')
#             if gamma is None:
#                 dset[...] = linear_kernel(x, x)
#             elif isinstance(gamma, numbers.Number):
#                 dset[...] = rbf_kernel(x, x, gamma=gamma)
#             else:
#                 raise ValueError
#             return dset[...]
#         else:
#             return f[key][...]

@cached("cache/gram.h5")
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
