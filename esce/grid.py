import numpy as np
from esce.util import load_grid_file

def logrange(start, stop, step=1., base=2.):
    base = float(base)
    return np.power(base, np.arange(start, stop + step, step))

grid = lambda n: {
    'lda': {},
    'logit': {'C': logrange(-20, 10, n)},
    'forest': {'n_estimators': [1000, ],
               'max_features': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0],
               'max_depth': [4, 6, 8, 10, 10000]},
    'ols': {},
    'lasso': {'alpha': logrange(-15, 15, n)},
    'ridge': {'alpha': logrange(-15, 15, n)},
    'svm-linear': {'C': logrange(-20, 10, n)},
    'svm-rbf': {'C': logrange(-10, 20, n), 'gamma': logrange(-25, 5, n)},
    'svm-sigmoid': {'C': logrange(-10, 20, n), 'gamma': logrange(-25, 5, n), 'coef0': [-1, 0, 1]},
    'svm-polynomial': {'C': logrange(-10, 20, n), 'gamma': logrange(-25, 5, n), 'coef0': [-1, 0, 1], 'degree': [2, 3]},
}

GRID = {
    "fine": grid(1),
    "default": grid(2),
    "coarse": grid(4)
}

def load_grid(grid_name):
    """
    Loads a grid from a name or a file.
    Valid names are fine, default and coarse.
    Grid files are required to be in YAML format.

    Arguments:
        grid_name: Name of path of the grid
    """
    if grid_name in ["fine", "default", "coarse"]:
        grid = GRID[grid_name]
    else:
        grid = load_grid_file(grid_name)
    return grid