from unittest import TestCase
from esce.data import get_mnist
from esce.sampling import split_grid
import numpy as np

class TestSampling(TestCase):
    def test_splits(self):
        y = np.random.choice([0,1], size=(1000,), p=[2./3, 1./3])
        num_seeds = 10
        samples = (50,100,200)

        splits = split_grid(y, num_seeds, samples, n_val=10, n_test=10)
        self.assertTrue(len(splits) == len(samples))
        for s in samples:
            self.assertTrue(len(splits[s]) == num_seeds)
