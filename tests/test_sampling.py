"""This module provides unit tests for splitting / sampling."""

import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid

from esce.grid import GRID
from esce.models import MODELS
from esce.sampling import split_grid


class TestSampling(TestCase):
    """Provide test cases for generating splits."""

    def test_splits(self):
        """Test split generation and verify number of seeds."""
        y = np.random.choice([0, 1], size=(1000,), p=[2.0 / 3, 1.0 / 3])
        num_seeds = 10
        samples = (50, 100, 200)

        splits = split_grid(y, num_seeds, samples, n_val=10, n_test=10)
        self.assertTrue(len(splits) == len(samples))
        for s in samples:
            self.assertTrue(len(splits[s]) == num_seeds)


class TestExample(TestCase):
    """Provide test cases for examples."""

    def test_example(self):
        """Test split generation and verify using a simple dataset."""
        warnings.simplefilter("ignore", category=DeprecationWarning)
        X, y = make_classification(
            n_samples=1000,
            n_features=2,
            n_redundant=0,
            n_informative=1,
            n_clusters_per_class=1,
            random_state=0,
        )
        n_seeds = 10
        splits = split_grid(y, n_seeds=n_seeds, n_val=100, n_test=100, do_stratify=True)
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
