import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid


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

        y_hat_train = model.predict(x[idx_train])
        y_hat_val = model.predict(x[idx_val])
        y_hat_test = model.predict(x[idx_test])
        return self.compute_metrics(
            y_hat_train, y_hat_val, y_hat_test, y[idx_train], y[idx_val], y[idx_test]
        )

    @abstractmethod
    def compute_metrics(
        self,
        y_hat_train: np.ndarray,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        pass


class ClassifierModel(BaseModel):
    """Base class for classifier models."""

    def compute_metrics(
        self,
        y_hat_train: np.ndarray,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        # Train score
        acc_train = accuracy_score(y_train, y_hat_train)
        f1_train = f1_score(y_train, y_hat_train, average="weighted")

        # Val score
        acc_val = accuracy_score(y_val, y_hat_val)
        f1_val = f1_score(y_val, y_hat_val, average="weighted")

        # Test score
        acc_test = accuracy_score(y_test, y_hat_test)
        f1_test = f1_score(y_test, y_hat_test, average="weighted")

        return {
            "acc_train": acc_train,
            "acc_val": acc_val,
            "acc_test": acc_test,
            "f1_train": f1_train,
            "f1_val": f1_val,
            "f1_test": f1_test,
        }


class RegressionModel(BaseModel):
    """Base class for regression models."""

    def compute_metrics(
        self,
        y_hat_train: np.ndarray,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        # Train score
        r2_train = r2_score(y_train, y_hat_train)
        mae_train = mean_absolute_error(y_train, y_hat_train)
        mse_train = mean_squared_error(y_train, y_hat_train)

        # Val score
        r2_val = r2_score(y_val, y_hat_val)
        mae_val = mean_absolute_error(y_val, y_hat_val)
        mse_val = mean_squared_error(y_val, y_hat_val)

        # Test score
        r2_test = r2_score(y_test, y_hat_test)
        mae_test = mean_absolute_error(y_test, y_hat_test)
        mse_test = mean_squared_error(y_test, y_hat_test)

        return {
            "r2_train": r2_train,
            "r2_val": r2_val,
            "r2_test": r2_test,
            "mae_train": mae_train,
            "mae_val": mae_val,
            "mae_test": mae_test,
            "mse_train": mse_train,
            "mse_val": mse_val,
            "mse_test": mse_test,
        }


MODELS = {
    "majority-classifier": ClassifierModel(
        lambda **args: DummyClassifier(strategy="most_frequent", **args),
        "majority classifier",
    ),
    "ridge-cls": ClassifierModel(
        lambda **args: RidgeClassifier(**args), "ridge classifier"
    ),
    "ridge-reg": RegressionModel(lambda **args: Ridge(**args), "ridge regressor"),
}


def get_existing_scores(scores_path_list):
    df_list = []
    for filename in scores_path_list:
        if os.stat(filename).st_size > 0:
            df_list.append(pd.read_csv(filename, index_col=False))

    if df_list:
        return pd.concat(
            df_list,
            axis=0,
            ignore_index=True,
        )
    else:
        return pd.DataFrame()


def fit(
    features_path,
    targets_path,
    split_path,
    scores_path,
    model_name,
    grid_path,
    existing_scores_path_list,
):
    split = json.load(open(split_path, "r"))
    if "error" in split:
        Path(scores_path).touch()
        return

    x = np.load(features_path)
    y = np.load(targets_path)
    grid = yaml.safe_load(open(grid_path, "r"))
    model = MODELS[model_name]

    df_existing_scores = get_existing_scores(existing_scores_path_list)

    scores = []
    for params in ParameterGrid(grid[model_name]):
        if (
            not df_existing_scores.empty
            and not df_existing_scores.loc[
                (df_existing_scores[list(params)] == pd.Series(params)).all(axis=1)
            ].empty
        ):
            score = dict(
                df_existing_scores.loc[
                    (df_existing_scores[list(params)] == pd.Series(params)).all(axis=1)
                ].iloc[0]
            )
            # print("retreived score", score)
        else:
            score = model.score(
                x,
                y,
                idx_train=split["idx_train"],
                idx_val=split["idx_val"],
                idx_test=split["idx_test"],
                **params
            )
            score.update(params)
            score.update({"n": split["samplesize"], "s": split["seed"]})
            # print("computed score", score)

        scores.append(score)

    pd.DataFrame(scores).to_csv(scores_path, index=None)


assert snakemake.wildcards.model in MODELS, "model not found"
fit(
    snakemake.input.features,
    snakemake.input.targets,
    snakemake.input.split,
    snakemake.output.scores,
    snakemake.wildcards.model,
    snakemake.input.grid,
    snakemake.params.existing_scores,
)
