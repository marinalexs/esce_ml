"""This module provides models and the sample complexity estimation code."""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import json
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge, RidgeClassifier)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.metrics.pairwise import (linear_kernel, polynomial_kernel,
                                      rbf_kernel, sigmoid_kernel)
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC, SVR


class BaseModel(ABC):
    """Base model class for each model."""

    def __init__(self, model_generator: Callable[..., Any], model_name: str) -> None:
        """Initialize class using a model that is initialized later."""
        self.model_generator = model_generator
        self.model_name = model_name

    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):  # type: ignore
        """Provide a score for the model performance on the data."""
        model = self.model_generator(**kwargs)
        model.fit(x[idx_train], y[idx_train])

        y_hat_val = model.predict(x[idx_val])
        y_hat_test = model.predict(x[idx_test])
        return self.compute_metrics(y_hat_val, y_hat_test, y[idx_val], y[idx_test])

    @abstractmethod
    def compute_metrics(
        self,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        pass


class ClassifierModel(BaseModel):
    """Base class for classifier models."""

    def compute_metrics(
        self,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute the classifier metrics.

        Arguments:
            y_hat_val: Generated labels by the classifier for the validation set
            y_hat_test: enerated labels by the classifier for the test set
            y_val: Ground truth labels of the validation set
            y_test: Ground truth labels of the test set

        Returns:
            Dictionary of scores containing accuracy and f1
        """
        # Val score
        acc_val = accuracy_score(y_val, y_hat_val)
        f1_val = f1_score(y_val, y_hat_val, average="weighted")

        # Test score
        acc_test = accuracy_score(y_test, y_hat_test)
        f1_test = f1_score(y_test, y_hat_test, average="weighted")

        return {
            "acc_val": acc_val,
            "acc_test": acc_test,
            "f1_val": f1_val,
            "f1_test": f1_test,
        }


class RegressionModel(BaseModel):
    """Base class for regression models."""

    def compute_metrics(
        self,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute regression metrics.

        Arguments:
            y_hat_val: Generated labels by the classifier for the validation set
            y_hat_test: enerated labels by the classifier for the test set
            y_val: Ground truth labels of the validation set
            y_test: Ground truth labels of the test set

        Returns:
            Dictionary of scores containing r2, mae and mse
        """
        # Val score
        r2_val = r2_score(y_val, y_hat_val)
        mae_val = mean_absolute_error(y_val, y_hat_val)
        mse_val = mean_squared_error(y_val, y_hat_val)

        # Test score
        r2_test = r2_score(y_test, y_hat_test)
        mae_test = mean_absolute_error(y_test, y_hat_test)
        mse_test = mean_squared_error(y_test, y_hat_test)

        return {
            "r2_val": r2_val,
            "r2_test": r2_test,
            "mae_val": mae_val,
            "mae_test": mae_test,
            "mse_val": mse_val,
            "mse_test": mse_test,
        }


def fit(features_path, targets_path, split_path, scores_path, model_name, grid_path):
    split = json.load(open(split_path, 'r'))
    if 'error' in split:
        Path(scores_path).touch()
        return

    x = np.genfromtxt(features_path, delimiter=',')
    y = np.genfromtxt(targets_path, delimiter=',')
    grid = yaml.safe_load(open(grid_path, 'r'))
    model = MODELS[model_name]

    scores = []
    # fixme: check which scores already exist from other grids
    for params in ParameterGrid(grid[model_name]):
        score = model.score(
            x, y, idx_train=split['idx_train'], idx_val=split['idx_val'], idx_test=split['idx_test'], **params)
        score.update(params)
        score.update({'n': split['samplesize'], 's': split['seed']})
        scores.append(score)

    pd.DataFrame(scores).to_csv(scores_path, index=None)


MODELS = {
    "majority-classifier": ClassifierModel(
        lambda **args: DummyClassifier(strategy="most_frequent",
                                       **args), 'majority classifier'
    ),
    "ridge": ClassifierModel(
        lambda **args: RidgeClassifier(**args), 'ridge classifier'
    ),

}


assert snakemake.wildcards.model in MODELS, 'model not found'
fit(snakemake.input.features, snakemake.input.targets, snakemake.input.split,
    snakemake.output.scores, snakemake.wildcards.model, snakemake.params.grid)
