import csv
import math
from abc import ABC, abstractmethod
from enum import Enum
from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

from esce.util import hash_dict
import esce


class KernelType(Enum):
    LINEAR = 1
    RBF = 2
    SIGMOID = 3
    POLYNOMIAL = 4


GRAM_PATH = Path("cache/gram.h5")


def get_gram_triu_key(
    x: np.ndarray,
    kernel: KernelType = KernelType.LINEAR,
    gamma: float = 0,
    coef0: float = 0,
    degree: float = 0,
) -> str:
    return f"/{md5(x).hexdigest()}/{kernel}_{gamma}_{coef0}_{degree}"


def compute_gram_matrix(
    x: np.ndarray,
    kernel: KernelType = KernelType.LINEAR,
    gamma: float = 0,
    coef0: float = 0,
    degree: float = 0,
) -> np.ndarray:
    if kernel == KernelType.LINEAR:
        return linear_kernel(x, x)
    elif kernel == KernelType.RBF:
        return rbf_kernel(x, x, gamma=gamma)
    elif kernel == KernelType.SIGMOID:
        return sigmoid_kernel(x, x, gamma=gamma, coef0=coef0)
    elif kernel == KernelType.POLYNOMIAL:
        return polynomial_kernel(x, x, degree=degree, gamma=gamma, coef0=coef0)
    else:
        raise ValueError


def get_gram_triu(
    x: np.ndarray,
    kernel: KernelType = KernelType.LINEAR,
    gamma: float = 0,
    coef0: float = 0,
    degree: float = 0,
) -> np.ndarray:
    """Calculates the upper triangle of the gram matrix.

    Args:
        data: Data to compute gram matrix of.
        kernel: Kernel type
        gamma: RBF kernel gamma.

    Returns:
        One-dimensional array containing the upper triangle
        of the computed gram matrix.
    """
    key = get_gram_triu_key(x, kernel, gamma, coef0, degree)
    with h5py.File(GRAM_PATH, "r") as f:
        if key in f:
            return f[key][...]

    with h5py.File(GRAM_PATH, "a") as f:
        K = compute_gram_matrix(x, kernel, gamma, coef0, degree)
        res = K[np.triu_indices(K.shape[0])]
        f.create_dataset(
            key,
            res.shape,
            dtype="f",
            data=res,
            **hdf5plugin.Blosc(
                cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE
            ),
        )
        return res


def get_gram(
    data: np.ndarray,
    kernel: KernelType = KernelType.LINEAR,
    gamma: float = 0,
    coef0: float = 0,
    degree: float = 0,
    cache: bool = False,
) -> np.ndarray:
    """Reconstructs the gram matrix based on upper triangle.

    Args:
        data: Data to compute gram matrix of.
        kernel: Kernel type
        gamma: Kernel gamma for RBF/sigmoid/polynomial
        coef0: Coefficient for sigmoid/polynomial
        degree: Degree of polynomial kernel
        cache: Whether to cache / use the cached kernel

    Returns:
        Two-dimensional gram matrix of the data.
    """
    x = data.astype(np.float32)
    if cache:
        tri = get_gram_triu(x, kernel, gamma, coef0, degree)
        n = int(0.5 * (math.sqrt(8 * len(tri) + 1) - 1))
        K = np.zeros((n, n), dtype=np.float32)
        K[np.triu_indices(n)] = tri
        K = K + K.T - np.diag(np.diag(K))
    else:
        K = compute_gram_matrix(x, kernel, gamma, coef0, degree)
    return K


class BaseModel(ABC):
    @abstractmethod
    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):  # type: ignore
        pass

    def compute_clf_metrics(
        self,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
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

    def order(self, param_grid: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        return param_grid

    def compute_regr_metrics(
        self,
        y_hat_val: np.ndarray,
        y_hat_test: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
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


class ClassifierModel(BaseModel):
    def __init__(self, model_generator: Callable[..., Any]) -> None:
        self.model_generator = model_generator

    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):  # type: ignore
        model = self.model_generator(**kwargs)
        model.fit(x[idx_train], y[idx_train])

        y_hat_val = model.predict(x[idx_val])
        y_hat_test = model.predict(x[idx_test])
        return self.compute_clf_metrics(y_hat_val, y_hat_test, y[idx_val], y[idx_test])


class RegressionModel(BaseModel):
    def __init__(self, model_generator: Callable[..., Any]) -> None:
        self.model_generator = model_generator

    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):  # type: ignore
        model = self.model_generator(**kwargs)
        model.fit(x[idx_train], y[idx_train])

        y_hat_val = model.predict(x[idx_val])
        y_hat_test = model.predict(x[idx_test])
        return self.compute_regr_metrics(y_hat_val, y_hat_test, y[idx_val], y[idx_test])


class KernelSVMModel(BaseModel):
    curr_config: Optional[Tuple[float, float, float]] = None
    cached_gram: np.ndarray
    cache: bool = False

    def __init__(self, kernel: KernelType = KernelType.LINEAR):
        self.kernel = kernel
        self.cached_gram = None

    def get_gram(self, x: np.ndarray, config: Tuple[float, float, float]) -> np.ndarray:
        if self.curr_config == config:
            return self.cached_gram
        else:
            gamma, coef0, degree = config
            self.cached_gram = get_gram(
                x,
                kernel=self.kernel,
                gamma=gamma,
                coef0=coef0,
                degree=degree,
                cache=self.cache,
            )
            self.curr_config = config
            return self.cached_gram

    def order(self, param_grid: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        if self.kernel == KernelType.RBF:
            return sorted(param_grid, key=lambda d: d["gamma"])  # type: ignore
        elif self.kernel == KernelType.SIGMOID:
            return sorted(param_grid, key=lambda d: (d["gamma"], d["coef0"]))
        elif self.kernel == KernelType.POLYNOMIAL:
            return sorted(
                param_grid, key=lambda d: (d["gamma"], d["coef0"], d["degree"])
            )
        else:
            return param_grid

    def score(  # type: ignore
        self, x, y, idx_train, idx_val, idx_test, C=1, gamma=0, coef0=0, degree=0
    ):
        gram = self.get_gram(x, (gamma, coef0, degree))
        model = SVC(C=C, kernel="precomputed", max_iter=10000)

        # Fit on train
        gram_ = gram[np.ix_(idx_train, idx_train)]
        model.fit(gram_, y[idx_train])

        # Val score
        gram_ = gram[np.ix_(idx_val, idx_train)]
        y_hat_val = model.predict(gram_)

        # Test score
        gram_ = gram[np.ix_(idx_test, idx_train)]
        y_hat_test = model.predict(gram_)

        return self.compute_clf_metrics(y_hat_val, y_hat_test, y[idx_val], y[idx_test])


def precompute_kernels(
    x: np.ndarray, models: Dict[str, BaseModel], grid: Dict[str, Dict[str, np.ndarray]]
) -> None:
    setup_cache_file()
    required = ["gamma", "coef0", "degree"]
    for model_name, model in models.items():
        if isinstance(model, KernelSVMModel):
            print(f"Precomputing {model_name}...")
            for params in ParameterGrid(grid[model_name]):
                print(f"=> Parameters: {params}")

                params = {k: v for k, v in params.items() if k in required}
                x = x.astype(np.float32)

                key = get_gram_triu_key(x, model.kernel, **params)
                print("key:", key)
                found = False
                with h5py.File(GRAM_PATH, "r") as f:
                    found = key in f
                if not found:
                    get_gram_triu(x, model.kernel, **params)


def setup_cache_file() -> None:
    GRAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GRAM_PATH.is_file():
        with h5py.File(GRAM_PATH, "w") as f:
            f.attrs["version"] = esce.__version__


def score_splits(
    outfile: Path,
    x: np.ndarray,
    y: np.ndarray,
    models: Dict[str, BaseModel],
    grid: Dict[str, Dict[str, np.ndarray]],
    splits: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    seeds: List[int],
    warm_start: bool = False,
    cache: bool = False,
) -> None:
    columns = [
        "model",
        "n",
        "s",
        "params",
        "param_hash",
        "acc_val",
        "acc_test",
        "f1_val",
        "f1_test",
        "r2_val",
        "r2_test",
        "mae_val",
        "mae_test",
        "mse_val",
        "mse_test",
    ]
    col2idx = {c: i for i, c in enumerate(columns)}
    if cache:
        setup_cache_file()
        for model in models.values():
            if isinstance(model, KernelSVMModel):
                model.cache = True

    # Read / Write results file
    if outfile.is_file() and warm_start:
        df = pd.read_csv(outfile, index_col=False)
    else:
        with outfile.open("w") as f:
            f.write(",".join(columns) + "\n")
        df = pd.read_csv(outfile, index_col=False)

    # Append results to csv file
    with outfile.open("a") as f:
        csvwriter = csv.writer(f, delimiter=",")

        for model_name, model in models.items():
            for params in model.order(ParameterGrid(grid[model_name])):
                param_hash = hash_dict(params)

                # For the n splis, only select n_seeds
                for n in splits:
                    for s in seeds:
                        idx_train, idx_val, idx_test = splits[n][s]

                        # Check if there is already an entry
                        # for the model, train size, seed and parameter combination
                        if not (
                            (df["model"] == model_name)
                            & (df["s"] == s)
                            & (df["n"] == n)
                            & (df["param_hash"] == param_hash)
                        ).any():
                            scores = model.score(
                                x, y, idx_train, idx_val, idx_test, **params
                            )  # type: ignore

                            row = [np.nan] * (len(columns) - 1)
                            row[:3] = [model_name, n, s, params, param_hash]
                            for k, v in scores.items():
                                row[col2idx[k]] = v

                            # Removes NaNs, prints scores
                            print(" ".join([str(r) for r in row if r == r]))

                            csvwriter.writerow(row)
                            f.flush()


MODELS = {
    "lda": ClassifierModel(
        lambda **args: LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", **args
        )
    ),
    "logit": ClassifierModel(
        lambda **args: LogisticRegression(solver="lbfgs", max_iter=100, **args)
    ),
    "forest": ClassifierModel(RandomForestClassifier),
    "ols": RegressionModel(LinearRegression),
    "lasso": RegressionModel(Lasso),
    "ridge": RegressionModel(Ridge),
    "svm-linear": KernelSVMModel(kernel=KernelType.LINEAR),
    "svm-rbf": KernelSVMModel(kernel=KernelType.RBF),
    "svm-sigmoid": KernelSVMModel(kernel=KernelType.SIGMOID),
    "svm-polynomial": KernelSVMModel(kernel=KernelType.POLYNOMIAL),
}

MODEL_NAMES = {
    "lda": "Linear Discriminant Analysis",
    "logit": "Logistic Regression",
    "forest": "Random Forest Classifier",
    "ols": "Ordinary Least Squared",
    "lasso": "Lasso Regression",
    "ridge": "Ridge Regression",
    "svm-linear": "Support Vector Machine (Linear)",
    "svm-rbf": "Support Vector Machine (RBF)",
    "svm-sigmoid": "Support Vector Machine (Sigmoid)",
    "svm-polynomial": "Support Vector Machine (Polynomial)",
}
