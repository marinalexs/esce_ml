from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def split(
    y: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates train, validation and test indices for the data.

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
    idx = np.arange(len(y))

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
    return idx_train, idx_val, idx_test


def split_grid(
    y: np.ndarray,
    n_seeds: int,
    n_samples: Iterable[int] = (100, 200, 500),
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
) -> Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Generated train, validation and test indices
    for a specified number of seeds and samples.

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
                )
            )
    return splits
