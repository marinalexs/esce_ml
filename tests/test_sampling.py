from unittest import TestCase
from esce.data import get_mnist
from esce.sampling import split_grid
from esce.models import fast_rbf, score_splits, MODELS
from esce.grid import GRID
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import ParameterGrid
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification
import warnings
import pandas as pd

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

class TestExample(TestCase):
    def test_example(self):
        warnings.simplefilter("ignore", category=DeprecationWarning)
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                n_informative=1, n_clusters_per_class=1, random_state=0)
        n_seeds = 10
        splits = split_grid(y, n_seeds=n_seeds, n_val=100, n_test=100)
        model = MODELS["logit"]
        grid = GRID["default"]
        scores = []
        for params in model.order(ParameterGrid(grid["logit"])):
            params["random_state"] = 0
            for n in splits:
                for s in range(n_seeds):
                    idx_train, idx_val, idx_test = splits[n][s]
                    score = model.score(X, y, idx_train, idx_val, idx_test, **params)
                    scores.append(score)
        df = pd.DataFrame(scores)
        self.assertTrue(df["acc_test"].mean() > 0.95)
        
