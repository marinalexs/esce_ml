import numpy as np

def logrange(start, stop, step=1., base=2.):
    base = float(base)
    return np.power(base, np.arange(start, stop + step, step))

grid = lambda n: {
    'ols': dict(),
    'lasso': {'alpha': logrange(-15, 15, n)},
    'ridge': {'alpha': logrange(-15, 15, n)},
    'svm-linear': {'C': logrange(-20, 10, n)},
    'svm-rbf': {'C': logrange(-10, 20, n), 'gamma': logrange(-25, 5, n)},
}

GRID = {
    "fine": grid(1),
    "default": grid(2),
    "coarse": grid(4)
}