from unittest import TestCase
from esce.data import get_mnist
from esce.sampling import split_grid
from esce.models import fast_rbf
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from numpy.testing import assert_array_equal

class TestSampling(TestCase):
    def test_splits(self):
        y = np.random.choice([0,1], size=(1000,), p=[2./3, 1./3])
        num_seeds = 10
        samples = (50,100,200)

        splits = split_grid(y, num_seeds, samples, n_val=10, n_test=10)
        self.assertTrue(len(splits) == len(samples))
        for s in samples:
            self.assertTrue(len(splits[s]) == num_seeds)

class TestAlgorithm(TestCase):
    def test_rbf_triu(self):
        gamma = 0.4
        for k in range(0, 4):
            for i in range(2,4):
                X, y = make_blobs(n_samples=4, centers=3, n_features=i, random_state=k)

                triu_custom = fast_rbf(X, gamma)
                triu_sk = rbf_kernel(X, X, gamma=gamma)[np.triu_indices(len(X),0)].astype(np.float32)
                assert_array_equal(triu_custom, triu_sk)