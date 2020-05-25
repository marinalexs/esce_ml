import sys

import numpy

from data import get_mnist
from models import score_splits
from sampling import split_grid


def run(method, n_components, noise=None, seeds=20):
    x, y = get_mnist(method=method, n_components=n_components, noise=noise)

    splits = split_grid(y, n_samples=(50, 100, 200, 500, 1000, 2000, 5000, 10000), n_seeds=seeds)

    results = score_splits(x, y, splits)

    results.to_csv(f'results/{method}_{n_components}_{noise}.csv')


TODO = []

# for sigma in numpy.arange(0.05, 1.05, 0.05):
#     TODO.append(['pca', 8, numpy.sqrt(sigma)])
#
# for sigma in [0, 0.5, 1, 2, 3, 5]:
#     TODO.append(['pca', 8, sigma])
#     TODO.append(['rp', 32, sigma])
#     TODO.append(['rp', 128, sigma])

for n_components in range(2, 10):
    TODO.append(['pca', 2**n_components])
    TODO.append(['rp', 2**n_components])

n = sys.argv[1]
args = TODO[int(n)]
print(args)
run(*args)