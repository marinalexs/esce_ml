import shelve
import numpy
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from joblib import hash
import numbers
import pandas as pd
import numpy as np
import math
from enum import Enum
from sklearn.model_selection import ParameterGrid
from abc import ABC, abstractmethod
import csv
from pathlib import Path
import pickle

from esce.util import cached
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_absolute_error, mean_squared_error

class KernelType(Enum):
    LINEAR = 1
    RBF = 2

@cached("cache/gram.h5")
def get_gram_triu(data, kernel=KernelType.LINEAR, gamma=0):
    """Calculates the upper triangle of the gram matrix.

    Args:
        data: Data to compute gram matrix of.
        kernel: Kernel type
        gamma: RBF kernel gamma.

    Returns:
        One-dimensional array containing the upper triangle
        of the computed gram matrix.
    """
    x = data.astype(np.float32)
    if kernel == KernelType.LINEAR:
        K = linear_kernel(x, x)
    elif kernel == KernelType.RBF:
        K = rbf_kernel(x, x, gamma=gamma)
    else:
        raise ValueError
    return K[np.triu_indices(K.shape[0])]

def get_gram(data, kernel=KernelType.LINEAR, gamma=0):
    """Reconstructs the gram matrix based on upper triangle.

    Args:
        data: Data to compute gram matrix of.
        gamma: RBF kernel gamma.

    Returns:
        Two-dimensional gram matrix of the data.
    """
    tri = get_gram_triu(data, kernel, gamma)
    n = int(0.5 * (math.sqrt(8 * len(tri) + 1) - 1))
    K = np.zeros((n,n), dtype=np.float32)
    K[np.triu_indices(n)] = tri

    # TODO: make this more efficient memory-wise?
    K = K + K.T - np.diag(np.diag(K))
    return K

class BaseModel(ABC):
    @abstractmethod
    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):
        pass

    def compute_clf_metrics(self, y_hat_val, y_hat_test, y_val, y_test):
        # Val score
        acc_val = accuracy_score(y_val, y_hat_val)
        f1_val = f1_score(y_val, y_hat_val, average="weighted")

        # Test score
        acc_test = accuracy_score(y_test, y_hat_test)
        f1_test = f1_score(y_test, y_hat_test, average="weighted")

        return {"acc_val": acc_val,
            "acc_test": acc_test,
            "f1_val": f1_val,
            "f1_test": f1_test }

    def compute_regr_metrics(self, y_hat_val, y_hat_test, y, idx_val, idx_test):
        # Val score
        r2_val = r2_score(y[idx_val], y_hat_val)
        mae_val = mean_absolute_error(y[idx_val], y_hat_val)
        mse_val = mean_squared_error(y[idx_val], y_hat_val)

        # Test score
        r2_test = r2_score(y[idx_test], y_hat_test)
        mae_test = mean_absolute_error(y[idx_test], y_hat_test)
        mse_test = mean_squared_error(y[idx_test], y_hat_test)

        return { "r2_val": r2_val, 
            "r2_test": r2_test, 
            "mae_val": mae_val, 
            "mae_test": mae_test, 
            "mse_val": mse_val, 
            "mse_test": mse_test }

class ClassifierModel(BaseModel):
    def __init__(self, model_generator):
        self.model_generator = model_generator

    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):
        model = self.model_generator(**kwargs)
        model.fit(x[idx_train], y[idx_train])

        y_hat_val = model.predict(x[idx_val])
        y_hat_test = model.predict(x[idx_test])
        return self.compute_clf_metrics(y_hat_val, y_hat_test, y[idx_val], y[idx_test])

class RegressionModel(BaseModel):
    def __init__(self, model_generator):
        self.model_generator = model_generator

    def score(self, x, y, idx_train, idx_val, idx_test, **kwargs):
        model = self.model_generator(**kwargs)
        model.fit(x[idx_train], y[idx_train])

        y_hat_val = model.predict(x[idx_val])
        y_hat_test = model.predict(x[idx_test])
        return self.compute_regr_metrics(y_hat_val, y_hat_test, y[idx_val], y[idx_test])

class KernelSVMModel(BaseModel):
    def __init__(self, kernel=KernelType.LINEAR):
        self.kernel = kernel
        self.prev_gamma = None
        self.cached_gram = None

    def get_gram(self, x, gamma):
        if self.prev_gamma == gamma:
            return self.cached_gram
        else:
            self.prev_gamma = gamma
            self.cached_gram = get_gram(x, kernel=self.kernel, gamma=gamma)
            return self.cached_gram

    def score(self, x, y, idx_train, idx_val, idx_test, C=1, gamma=0):
        gram = self.get_gram(x, gamma)
        model = SVC(C=C, kernel='precomputed', max_iter=1000)

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

def score_splits(outfile, x, y, models, grid, splits, seeds, warm_start=False):
    columns = ["model","n","s","params","param_hash",
        "acc_val","acc_test","f1_val","f1_test",
        "r2_val","r2_test","mae_val","mae_test","mse_val","mse_test"]
    col2idx = { c:i for i,c in enumerate(columns) }

    # Read / Write results file
    if outfile.is_file() and warm_start:
        df = pd.read_csv(outfile, index_col=False)
    else:
        with outfile.open("w") as f:
            f.write(','.join(columns)+"\n")
        df = pd.read_csv(outfile)

    # Append results to csv file
    with outfile.open("a") as f:
        csvwriter = csv.writer(f, delimiter=",")

        for model_name in models:
            model = models[model_name]

            # For the n splis, only select n_seeds
            for n in splits:
                for s in seeds:
                    idx_train, idx_val, idx_test = splits[n][s]

                    for params in ParameterGrid(grid[model_name]):
                        param_hash = hash(params)

                        # Check if there is already an entry
                        # for the model, train size, seed and parameter combination
                        if not ((df["model"] == model_name) & (df["s"] == s) & (df["n"] == n) & (df["param_hash"] == param_hash)).any():
                            scores = model.score(x, y, idx_train, idx_val, idx_test, **params)

                            row = [np.nan] * (len(columns)-1)
                            row[:3] = [model_name, n, s, params, param_hash]
                            for k,v in scores.items():
                                row[col2idx[k]] = v

                            # Removes NaNs, prints scores
                            print(' '.join([str(r) for r in row if r == r]))

                            csvwriter.writerow(row)
                            f.flush()

MODELS = {
    "lda": ClassifierModel(lambda **args: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', **args)),
    "logit": ClassifierModel(lambda **args: LogisticRegression(solver='lbfgs', max_iter=100, **args)),
    "forest": ClassifierModel(RandomForestClassifier),
    "ols": RegressionModel(LinearRegression),
    "lasso": RegressionModel(Lasso),
    "ridge": RegressionModel(Ridge),
    "svm-linear": KernelSVMModel(kernel=KernelType.LINEAR),
    "svm-rbf": KernelSVMModel(kernel=KernelType.RBF)
}