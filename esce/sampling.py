import numpy as np
from sklearn.model_selection import train_test_split

def split(y, n_train, n_val=1000, n_test=1000, do_stratify=True, seed=0):
    idx = np.arange(len(y))

    stratify = y if do_stratify else None
    idx_train, idx_test = train_test_split(idx, test_size=n_test, stratify=stratify, random_state=seed)

    stratify = y[idx_train] if do_stratify else None
    idx_train, idx_val = train_test_split(idx_train, train_size=n_train, test_size=n_val, stratify=stratify,
                                          random_state=seed)
    return idx_train, idx_val, idx_test

def split_grid(y, n_seeds, n_samples=(100, 200, 500), n_val=1000, n_test=1000, do_stratify=True):
    splits = {}
    for n in n_samples:
        splits[n] = {}
        for s in range(n_seeds):
            splits[n][s] = split(y, seed=s, n_train=n, n_val=n_val, n_test=n_test, do_stratify=do_stratify)
    return splits
