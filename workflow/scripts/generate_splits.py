"""This module prodives method to split and resample the data."""

from typing import Dict, Iterable, List, Tuple, Optional, Union

import numpy as np
import yaml, os
import pickle
import json
from sklearn.model_selection import train_test_split


def generate_random_split(
    y: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Optional[np.ndarray] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    split = {'idx_train': idx_originial[idx_train], 'idx_val': idx_originial[idx_val], 'idx_test': idx_originial[idx_test]}
    split = {k: v.tolist() for k, v in split.items()}  # for yaml
    split.update({'samplesize': n_train,
                    'seed': seed,
                    'stratify': do_stratify})  # metadata

    return split


def generate_matched_split(
    y: np.ndarray,
    match: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Optional[np.ndarray] = False,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    random_state = np.random.RandomState(seed)
    split = generate_random_split(y,n_train//2,n_val//2,n_test//2,do_stratify,seed, mask)
    idx_all = np.arange(len(y))

    mask[split['idx_train']]=False
    mask[split['idx_val']]=False
    mask[split['idx_test']]=False
    mask[y==1]=False

    matching_scores = []
    for idx_set in ['idx_train', 'idx_val', 'idx_test']:
        control_group = []
        for idx in split[idx_set]:
            idx_pool = idx_all[mask]
            scores = (match[mask]-match[idx])**2

            t=random_state.permutation(np.column_stack((scores,idx_pool)))
            t_idx = np.argmin(t.T[0])

            score_match = t.T[0][t_idx]
            matching_scores.append(score_match)

            idx_match = t.T[1][t_idx].astype(int)
            control_group.append(int(idx_match))
            mask[idx_match] = False

        split[idx_set]+=control_group

    split.update({'average_matching_score':np.mean(matching_scores)})
    split['samplesize']*=2

    return split
    

def write_splitfile(features_path, targets_path, split_path, matching_path, n_train, n_val, n_test, seed, stratify=False):
    x = np.genfromtxt(features_path, delimiter=',')
    x_mask = np.all(np.isfinite(x), 1)

    y = np.genfromtxt(targets_path, delimiter=',')
    y_mask = np.isfinite(y)

    xy_mask = np.logical_and(x_mask, y_mask)

    matching = np.genfromtxt(matching_path, delimiter=',')
    if len(matching)==len(y) and len(matching.shape)>1:
        m_mask = np.all(np.isfinite(matching), 1)
        xy_mask = np.logical_and(xy_mask,m_mask)
    elif len(matching)==len(y) and len(matching.shape)==1:
        m_mask = np.isfinite(matching)
        xy_mask = np.logical_and(xy_mask,m_mask)
    elif len(matching)==0:
        matching = False
    else:
        raise Exception('invalid matching file')

    if  matching is False and sum(xy_mask) > n_train + n_val + n_test:
        split_dict = generate_random_split(y=y, n_train=n_train, n_val=n_val,
                                    n_test=n_test, do_stratify=stratify, mask=xy_mask, seed=seed)
    elif sum(xy_mask[y==1]) > n_train//2 + n_val//2 + n_test//2:
        split_dict = generate_matched_split(y=y, match=matching, n_train=n_train, n_val=n_val,
                                    n_test=n_test, do_stratify=stratify, mask=xy_mask, seed=seed)
    else:
        split_dict = {'error': 'insufficient samples'}

    print(split_dict)
    with open(split_path, 'w') as f:
        json.dump(split_dict, f)


n_train = int(snakemake.wildcards.samplesize)
n_val = min(round(n_train*snakemake.config['val_test_frac']),snakemake.config['val_test_max'])
n_test = min(round(n_train*snakemake.config['val_test_frac']),snakemake.config['val_test_max'])

write_splitfile(features_path=snakemake.input.features, targets_path=snakemake.input.targets, split_path=snakemake.output.split, matching_path=snakemake.input.matching, n_train=n_train,
                n_val=n_val, n_test=n_test, seed=int(snakemake.wildcards.seed), stratify=int(snakemake.wildcards.stratify))
