"""
fit_model.py
====================================
This module defines base classes and utilities for model training and evaluation.
It includes abstract base classes for classifiers and regressors, specific model
implementations, and functions to fit models and record their performance metrics.
"""

import json
import os
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Union, Optional, Tuple

import numpy as np
import h5py
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.base import BaseEstimator

# Set up logging
log_level = os.environ.get('ESCE_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')


class BaseModel(ABC):
    """
    Abstract base model class for various machine learning models.
    
    This class provides a common interface for different types of models,
    including methods for scoring, scaling features/targets, and computing metrics.
    """

    scale_features: bool
    scale_targets: bool

    def __init__(self, model_generator: Callable[..., BaseEstimator], model_name: str):
        """
        Initialize the BaseModel with a model generator and name.

        Args:
            model_generator (Callable[..., BaseEstimator]): A callable that generates the model instance.
            model_name (str): Descriptive name of the model.
        """
        self.model_generator: Callable[..., BaseEstimator] = model_generator
        self.model_name: str = model_name

    def _check_data_validity(self, *arrays: np.ndarray) -> None:
        """
        Check for NaN or infinite values in the input arrays.

        Args:
            *arrays: Variable number of numpy arrays to check.

        Raises:
            ValueError: If any NaN or infinite values are found.
        """
        for arr in arrays:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise ValueError("Input data contains NaN or infinite values.")

    def score(
        self,
        x: h5py.Dataset,
        y: h5py.Dataset,
        cni: h5py.Dataset,
        idx_train: List[int],
        idx_val: List[int],
        idx_test: List[int],
        mode: Literal["normal", "with-cni", "only-cni"] = "normal",
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Evaluate the model's performance on training, validation, and test sets.

        Args:
            x (h5py.Dataset): Feature matrix dataset.
            y (h5py.Dataset): Target values dataset.
            cni (h5py.Dataset): Confounding variables dataset.
            idx_train (List[int]): Indices for the training set.
            idx_val (List[int]): Indices for the validation set.
            idx_test (List[int]): Indices for the test set.
            mode (Literal["normal", "with-cni", "only-cni"]): Mode of feature inclusion.
            **kwargs: Additional keyword arguments for the model generator.

        Returns:
            Dict[str, float]: Dictionary of computed performance metrics.
        """
        # Select features and targets based on the specified mode
        x_train, x_val, x_test, y_train, y_val, y_test = self._select_features_and_targets(
            x, y, cni, idx_train, idx_val, idx_test, mode
        )

        # Check for NaN or infinite values
        self._check_data_validity(x_train, x_val, x_test, y_train, y_val, y_test)

        # Initialize and fit the model
        model = self.model_generator(**kwargs)
        x_train_scaled, x_val_scaled, x_test_scaled = self._scale_features(x_train, x_val, x_test)
        y_train_scaled = self._scale_targets(y_train)
        model.fit(x_train_scaled, y_train_scaled)

        # Predict and evaluate
        y_hat_train = self._predict(model, x_train_scaled)
        y_hat_val = self._predict(model, x_val_scaled)
        y_hat_test = self._predict(model, x_test_scaled)

        return self.compute_metrics(
            y_hat_train, y_hat_val, y_hat_test, y_train, y_val, y_test
        )

    def _select_features_and_targets(
        self,
        x: h5py.Dataset,
        y: h5py.Dataset,
        cni: h5py.Dataset,
        idx_train: List[int],
        idx_val: List[int],
        idx_test: List[int],
        mode: Literal["normal", "with-cni", "only-cni"],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select features and targets based on the specified mode.

        Args:
            x (h5py.Dataset): Feature matrix dataset.
            y (h5py.Dataset): Target values dataset.
            cni (h5py.Dataset): Confounding variables dataset.
            idx_train (List[int]): Indices for the training set.
            idx_val (List[int]): Indices for the validation set.
            idx_test (List[int]): Indices for the test set.
            mode (Literal["normal", "with-cni", "only-cni"]): Mode of feature inclusion.

        Returns:
            tuple: Selected features and targets for train, validation, and test sets.
        """
        if mode == "normal":
            x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
        elif mode == "with-cni":
            x_train = np.concatenate([x[idx_train], cni[idx_train]], axis=1)
            x_val = np.concatenate([x[idx_val], cni[idx_val]], axis=1)
            x_test = np.concatenate([x[idx_test], cni[idx_test]], axis=1)
        elif mode == "only-cni":
            x_train, x_val, x_test = cni[idx_train], cni[idx_val], cni[idx_test]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _scale_features(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray
    ) -> tuple:
        """
        Scale features if required.

        Args:
            x_train (np.ndarray): Training features.
            x_val (np.ndarray): Validation features.
            x_test (np.ndarray): Test features.

        Returns:
            tuple: Scaled features for train, validation, and test sets.
        """
        if self.scale_features:
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.transform(x_val)
            x_test_scaled = scaler.transform(x_test)
        else:
            x_train_scaled, x_val_scaled, x_test_scaled = x_train, x_val, x_test
        return x_train_scaled, x_val_scaled, x_test_scaled

    def _scale_targets(self, y_train: np.ndarray) -> np.ndarray:
        """
        Scale targets if required.

        Args:
            y_train (np.ndarray): Training targets.

        Returns:
            np.ndarray: Scaled training targets.
        """
        if self.scale_targets:
            self.y_scaler = StandardScaler()
            y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = y_train
        return y_train_scaled

    def _predict(self, model: Any, x: np.ndarray) -> np.ndarray:
        """
        Make predictions and inverse transform if necessary.

        Args:
            model (Any): Fitted model instance.
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Predictions.
        """
        y_hat = model.predict(x)
        if self.scale_targets:
            y_hat = self.y_scaler.inverse_transform(y_hat.reshape(-1, 1)).flatten()
        return y_hat

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
        """
        Compute performance metrics based on predictions and true values.

        This method should be implemented by subclasses to provide
        model-specific performance metrics.

        Args:
            y_hat_train (np.ndarray): Predictions for training set.
            y_hat_val (np.ndarray): Predictions for validation set.
            y_hat_test (np.ndarray): Predictions for test set.
            y_train (np.ndarray): True values for training set.
            y_val (np.ndarray): True values for validation set.
            y_test (np.ndarray): True values for test set.

        Returns:
            Dict[str, float]: Dictionary of computed metrics.
        """
        pass


class ClassifierModel(BaseModel):
    """Base class for classification models."""

    scale_features = True
    scale_targets = False

    def compute_metrics(
        self,
        y_hat_train: np.ndarray,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute classification metrics: accuracy and F1-score.

        Args:
            y_hat_train (np.ndarray): Predictions for training set.
            y_hat_val (np.ndarray): Predictions for validation set.
            y_hat_test (np.ndarray): Predictions for test set.
            y_train (np.ndarray): True values for training set.
            y_val (np.ndarray): True values for validation set.
            y_test (np.ndarray): True values for test set.

        Returns:
            Dict[str, float]: Dictionary containing accuracy and F1-score for each dataset split.
        """
        return {
            "acc_train": accuracy_score(y_train, y_hat_train),
            "acc_val": accuracy_score(y_val, y_hat_val),
            "acc_test": accuracy_score(y_test, y_hat_test),
            "bal_acc_train": balanced_accuracy_score(y_train, y_hat_train),
            "bal_acc_val": balanced_accuracy_score(y_val, y_hat_val),
            "bal_acc_test": balanced_accuracy_score(y_test, y_hat_test),
            "f1_train": f1_score(y_train, y_hat_train, average="weighted"),
            "f1_val": f1_score(y_val, y_hat_val, average="weighted"),
            "f1_test": f1_score(y_test, y_hat_test, average="weighted"),
        }


class RegressionModel(BaseModel):
    """Base class for regression models."""

    scale_features = True
    scale_targets = True

    def compute_metrics(
        self,
        y_hat_train: np.ndarray,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute regression metrics: R², MAE, and MSE.

        Args:
            y_hat_train (np.ndarray): Predictions for training set.
            y_hat_val (np.ndarray): Predictions for validation set.
            y_hat_test (np.ndarray): Predictions for test set.
            y_train (np.ndarray): True values for training set.
            y_val (np.ndarray): True values for validation set.
            y_test (np.ndarray): True values for test set.

        Returns:
            Dict[str, float]: Dictionary containing R², MAE, and MSE for each dataset split.
        """
        return {
            "r2_train": r2_score(y_train, y_hat_train),
            "r2_val": r2_score(y_val, y_hat_val),
            "r2_test": r2_score(y_test, y_hat_test),
            "mae_train": mean_absolute_error(y_train, y_hat_train),
            "mae_val": mean_absolute_error(y_val, y_hat_val),
            "mae_test": mean_absolute_error(y_test, y_hat_test),
            "mse_train": mean_squared_error(y_train, y_hat_train),
            "mse_val": mean_squared_error(y_val, y_hat_val),
            "mse_test": mean_squared_error(y_test, y_hat_test),
        }


# Define available models with their corresponding generators and names
MODELS: Dict[str, Union[ClassifierModel, RegressionModel]] = {
    "majority-classifier": ClassifierModel(
        lambda **args: DummyClassifier(strategy="most_frequent", **args),
        "majority classifier",
    ),
    "mean-regressor": RegressionModel(
        lambda **args: DummyRegressor(strategy="mean", **args), "mean regressor"
    ),
    "ridge-cls": ClassifierModel(
        lambda **args: RidgeClassifier(**args), "ridge classifier"
    ),
    "ridge-reg": RegressionModel(lambda **args: Ridge(**args), "ridge regressor"),
}


def get_existing_scores(scores_path_list: List[str]) -> pd.DataFrame:
    """
    Aggregate existing scores from multiple score files into a single DataFrame.

    Args:
        scores_path_list (List[str]): List of file paths containing existing scores.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all existing scores.
    """
    df_list: List[pd.DataFrame] = [pd.read_csv(f) for f in scores_path_list if os.path.getsize(f) > 0]
    return pd.concat(df_list, axis=0, ignore_index=True) if df_list else pd.DataFrame()


def fit(
    features_path: str,
    targets_path: str,
    split_path: str,
    scores_path: str,
    model_name: str,
    grid: Dict[str, Any],
    existing_scores_path_list: List[str],
    confound_correction_method: str,
    cni_path: str,
) -> pd.DataFrame:
    """
    Fit a specified model to the data and record its performance metrics.

    This function loads data, performs model fitting, and saves the results.

    Args:
        features_path (str): Path to the features HDF5 file.
        targets_path (str): Path to the targets HDF5 file.
        split_path (str): Path to the JSON file containing data splits.
        scores_path (str): Path to save the computed scores.
        model_name (str): Name of the model to be fitted.
        grid (Dict[str, Any]): Hyperparameter grid for model tuning.
        existing_scores_path_list (List[str]): List of paths to existing score files.
        confound_correction_method (str): Method for confound correction.
        cni_path (str): Path to the confounding variables (CNI) HDF5 file.
    """
    logging.info(f"Starting model fitting for {model_name}")

    # Load the data split information
    with open(split_path, 'r') as f:
        split = json.load(f)
    if "error" in split:
        logging.warning(f"Error found in split file: {split['error']}")
        Path(scores_path).touch()
        return

    # Check if the model is valid
    if model_name not in MODELS:
        error_msg = f"Invalid model type: {model_name}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    model = MODELS[model_name]
    logging.info(f"Using model: {model.model_name}")

    df_existing_scores = get_existing_scores(existing_scores_path_list)
    logging.debug(f"Loaded {len(df_existing_scores)} existing scores")

    scores: List[Dict[str, Any]] = []
    with h5py.File(features_path, "r") as fx, h5py.File(targets_path, "r") as fy, h5py.File(cni_path, "r") as fc:
        x, cni = fx["data"], fc["data"]
        
        # Load target data
        y = fy["data"][:]
        
        # Ensure y is 2-dimensional
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Check for insufficient samples
        if len(split["idx_train"]) < 2 or len(split["idx_val"]) < 2 or len(split["idx_test"]) < 2:
            logging.warning("Insufficient samples for train/val/test split")
            Path(scores_path).touch()
            return

        # Iterate over all combinations of hyperparameters
        for params in ParameterGrid(grid[model_name]):
            logging.debug(f"Evaluating hyperparameters: {params}")
            
            # Check if we already have scores for this parameter combination
            df_existing_scores_filtered = df_existing_scores.loc[
                (df_existing_scores[list(params)] == pd.Series(params)).all(axis=1)
            ] if not df_existing_scores.empty else pd.DataFrame()

            if not df_existing_scores_filtered.empty:
                logging.info("Using existing scores for current parameter combination")
                score = dict(df_existing_scores_filtered.iloc[0])
            else:
                logging.info("Computing new scores for current parameter combination")
                score = model.score(
                    x, y, cni,
                    idx_train=split["idx_train"],
                    idx_val=split["idx_val"],
                    idx_test=split["idx_test"],
                    mode=confound_correction_method if confound_correction_method in ['with-cni', 'only-cni'] else 'normal',
                    **params,
                )
                score.update(params)
                score.update({"n": split["samplesize"], "s": split["seed"]})

            scores.append(score)

    # Save all scores to a CSV file
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(scores_path, index=None)
    logging.info(f"Saved scores to {scores_path}")
    return df_scores


if __name__ == "__main__":
    logging.info("Starting fit_model.py script")
    
    assert snakemake.wildcards.model in MODELS, f"Model '{snakemake.wildcards.model}' not found in predefined MODELS."

    result_df = fit(
        features_path=snakemake.input.features,
        targets_path=snakemake.input.targets,
        split_path=snakemake.input.split,
        scores_path=snakemake.output.scores,
        model_name=snakemake.wildcards.model,
        grid=snakemake.params.grid,
        existing_scores_path_list=snakemake.params.existing_scores,
        confound_correction_method=snakemake.wildcards.confound_correction_method,
        cni_path=snakemake.input.covariates,
    )
    
    logging.info("Completed fit_model.py script")