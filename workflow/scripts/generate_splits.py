"""This module prodives method to split and resample the data."""

from typing import Dict, Iterable, List, Tuple, Optional, Union

import numpy as np
import yaml
import pickle
import json
from sklearn.model_selection import train_test_split


def generate_split(
    y: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Optional[np.ndarray] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate train, validation and test indices for the data.

    Arguments:
        y: Data labels
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of testing samples
        do_stratify: If the splits should be sampled using stratified sampling
        seed: Random state seed

    Returns:
        Tuple of index lists (idx_train, idx_val, idx_test)
    """
    if mask is False:
        idx_originial = np.arange(len(y))
        idx = np.arange(len(y))
    else:
        idx_originial = np.arange(len(y))[mask]
        idx = np.arange(len(y[mask]))
        y = y[mask]

    stratify = y if do_stratify else None
    idx_train, idx_test = train_test_split(
        idx, test_size=n_test, stratify=stratify, random_state=seed
    )

    stratify = y[idx_train] if do_stratify else None
    idx_train, idx_val = train_test_split(
        idx_train,
        train_size=n_train,
        test_size=n_val,
        stratify=stratify,
        random_state=seed,
    )
    return {'idx_train': idx_originial[idx_train], 'idx_val': idx_originial[idx_val], 'idx_test': idx_originial[idx_test]}


def write_splitfile(features_path, targets_path, split_path, n_train, n_val, n_test, seed, stratify=False):
    x = np.genfromtxt(features_path, delimiter=',')
    x_mask = np.all(np.isfinite(x), 1)

    y = np.genfromtxt(targets_path, delimiter=',')
    y_mask = np.isfinite(y)

    xy_mask = np.logical_and(x_mask, y_mask)
    print(sum(xy_mask),  n_train + n_val + n_test)

    if sum(xy_mask) > n_train + n_val + n_test:
        split_dict = generate_split(y=y, n_train=n_train, n_val=n_val,
                                    n_test=n_test, do_stratify=stratify, mask=xy_mask, seed=seed)
        split_dict = {k: v.tolist() for k, v in split_dict.items()}  # for yaml
    else:
        split_dict = {'error': 'insufficient samples'}

    split_dict.update({'samplesize': n_train,
                       'seed': seed,
                       'stratify': stratify})  # metadata

    with open(split_path, 'w') as f:
        json.dump(split_dict, f)


# FIXME: implement output.matching
write_splitfile(features_path=snakemake.input.features, targets_path=snakemake.input.targets, split_path=snakemake.output.split, n_train=int(snakemake.wildcards.samplesize),
                n_val=snakemake.config['n_val'], n_test=snakemake.config['n_test'], seed=int(snakemake.wildcards.seed), stratify=int(snakemake.wildcards.stratify))
