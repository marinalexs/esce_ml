import numpy as np

def logrange(start, stop, step=1., base=2.):
    base = float(base)
    return np.power(base, np.arange(start, stop + step, step))

GRID = {
    'ols': dict(),
    'svm-linear': {'C': logrange(-20, 10, 1)},
    'svm-rbf': {'C': logrange(-10, 20, 1), 'gamma': logrange(-25, 5, 1)},
}
