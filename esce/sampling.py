"""This module prodives method to split and resample the data."""

from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split


def split(
    y: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Union[bool, np.ndarray] = False,
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
    return idx_originial[idx_train], idx_originial[idx_val], idx_originial[idx_test]


def split_grid(
    y: np.ndarray,
    n_seeds: int,
    n_samples: Iterable[int] = (100, 200, 500),
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    mask: Union[bool, np.ndarray] = False,
) -> Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Generate train, validation and test indices for the given seeds and samples.

    Arguments:
        y: Data lables
        n_seeds: Number of seeds to use
        n_samples: Which sample ticks to use
        n_val: Number of validation samples
        n_test: Number of testing samples
        do_stratify: Whether or not to use stratified sampling

    Returns:
        Dictionary of splits. Samples are the first key, seeds are the subkey.
    """
    splits: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for n in n_samples:
        # skip if not enough samples
        if n + n_val + n_test > len(y[mask]):
            print(f"skipping n={n}, not enough samples")
            continue

        splits[n] = []
        for s in range(n_seeds):
            splits[n].append(
                split(
                    y,
                    seed=s,
                    n_train=n,
                    n_val=n_val,
                    n_test=n_test,
                    do_stratify=do_stratify,
                    mask=mask,
                )
            )

    return splits
