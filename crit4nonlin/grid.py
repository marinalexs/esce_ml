import numpy


def logrange(start, stop, step=1., base=2.):
    base = float(base)
    return numpy.power(base, numpy.arange(start, stop + step, step))


GRID = {
    'linear': {'C': logrange(-20, 10, 1),  'gamma': [None, ]},
    'rbf': {'C': logrange(-10, 20, 1), 'gamma': logrange(-25, 5, 1)},
}
