from unittest import TestCase
from esce.models import MODELS
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

class TestModels(TestCase):
    def test_regression(self):
        model = MODELS["ridge"]
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                n_informative=1, n_clusters_per_class=1, random_state=0)
        idx = np.arange(len(y))
        idx_train, idx_test = train_test_split(idx, train_size=0.7, random_state=0)
        idx_train, idx_val = train_test_split(idx_train, train_size=0.7,random_state=0)
            
        score = model.score(X, y, idx_train, idx_val, idx_test, random_state=0)
        keys = { "r2_val", "r2_test", "mae_val","mae_test", "mse_val","mse_test" }
        self.assertEqual(score.keys(), keys)
        self.assertAlmostEqual(score["r2_val"], 0.75, places=2)
        self.assertAlmostEqual(score["r2_test"], 0.76, places=2)

    def test_classification(self):
        model = MODELS["lda"]
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                n_informative=1, n_clusters_per_class=1, random_state=0)
        idx = np.arange(len(y))
        idx_train, idx_test = train_test_split(idx, train_size=0.7, random_state=0)
        idx_train, idx_val = train_test_split(idx_train, train_size=0.7,random_state=0)
            
        score = model.score(X, y, idx_train, idx_val, idx_test)
        keys = {"acc_val", "acc_test", "f1_val", "f1_test" }
        self.assertEqual(score.keys(), keys)
        self.assertAlmostEqual(score["acc_val"], 0.94, places=2)
        self.assertAlmostEqual(score["acc_test"], 0.96, places=2)