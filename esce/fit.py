from pathlib import Path

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast, Union
import h5py
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from esce.models import MODELS

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

    with h5py.File(features_path, 'r') as f:
        x = f['data'][:]

    with h5py.File(targets_path, 'r') as f:
        y = f['data'][:]

    assert np.isfinite(x[split["idx_train"]]).all()
    assert np.isfinite(y[split["idx_train"]]).all()

    grid = yaml.safe_load(open(grid_path, "r"))
    model = MODELS[model_name]

    # if model is ClassifierModel, make sure that y is categorical
                    

    df_existing_scores = get_existing_scores(existing_scores_path_list)

    scores = []
    for params in ParameterGrid(grid[model_name]):
        df_existing_scores_filtered = lambda: df_existing_scores.loc[
            (df_existing_scores[list(params)] == pd.Series(params)).all(axis=1)
        ]
        if not df_existing_scores.empty and not df_existing_scores_filtered().empty:
            score = dict(df_existing_scores_filtered().iloc[0])
            # print("retrieved score", score)
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


