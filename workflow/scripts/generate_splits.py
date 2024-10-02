"""
This module provides functions for generating train/validation/test splits for datasets,
including options for stratification, matching, and confound correction.
"""

import json
from typing import Optional, Tuple, Dict, Union, List

import h5py
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""

    def default(self, obj):
        """Convert NumPy data types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_random_split(
    y: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Union[List[int], int, bool]]:
    """
    Generate a random train/validation/test split of the data.

    Args:
        y (np.ndarray): Target labels.
        n_train (int): Number of training samples.
        n_val (int, optional): Number of validation samples. Defaults to 1000.
        n_test (int, optional): Number of test samples. Defaults to 1000.
        do_stratify (bool, optional): Whether to perform stratified splitting. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        mask (Optional[np.ndarray], optional): Boolean mask to select a subset of data. Defaults to None.

    Returns:
        dict: Dictionary containing indices for train, validation, and test sets along with metadata.
    """
    if mask is None:
        idx_original = np.arange(len(y))
        idx = np.arange(len(y))
    else:
        idx_original = np.arange(len(y))[mask]
        idx = np.arange(len(y[mask]))
        y = y[mask]

    stratify = y if do_stratify else None

    # Split into training and testing sets
    idx_train, idx_test = train_test_split(
        idx, test_size=n_test, stratify=stratify, random_state=seed
    )

    # Update stratify based on training indices for further splitting
    stratify = y[idx_train] if do_stratify else None

    # Split training set into training and validation sets
    idx_train, idx_val = train_test_split(
        idx_train,
        train_size=n_train,
        test_size=n_val,
        stratify=stratify,
        random_state=seed,
    )

    split = {
        "idx_train": idx_original[idx_train].tolist(),  # Convert to list
        "idx_val": idx_original[idx_val].tolist(),      # Convert to list
        "idx_test": idx_original[idx_test].tolist(),    # Convert to list
        "samplesize": n_train,
        "seed": seed,
        "stratify": do_stratify,
    }

    return split


def generate_matched_split(
    y: np.ndarray,
    match: np.ndarray,
    n_train: int,
    n_val: int = 1000,
    n_test: int = 1000,
    do_stratify: bool = False,
    seed: int = 0,
    mask: Optional[np.ndarray] = False,
) -> Dict[str, Union[List[int], int, bool, float]]:
    """
    Generate a matched train/validation/test split of the data based on confounding variables.

    Args:
        y (np.ndarray): Target labels.
        match (np.ndarray): Confounding variables for matching.
        n_train (int): Number of training samples.
        n_val (int, optional): Number of validation samples. Defaults to 1000.
        n_test (int, optional): Number of test samples. Defaults to 1000.
        do_stratify (bool, optional): Whether to perform stratified splitting. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        mask (Optional[np.ndarray], optional): Boolean mask to select a subset of data. Defaults to False.

    Returns:
        dict: Dictionary containing indices for train, validation, and test sets along with metadata.
    """
    random_state = np.random.RandomState(seed)
    mask = mask.copy()
    mask_orig = mask.copy()

    # Generate initial random split for positive class
    split = generate_random_split(
        y=y,
        n_train=n_train // 2,
        n_val=n_val // 2,
        n_test=n_test // 2,
        do_stratify=do_stratify,
        seed=seed,
        mask=np.logical_and(mask, y == 1),
    )
    idx_all = np.arange(len(y))

    # Exclude positive class from matching pool
    mask[y == 1] = False
    assert np.isfinite(match[mask]).all(), "Confounding variables contain non-finite values."

    # Standardize confounding variables for matching
    match = StandardScaler().fit_transform(match)
    matching_scores = []

    for idx_set in ["idx_train", "idx_val", "idx_test"]:
        control_group = []  # Control group for each split set
        for idx in split[idx_set]:
            idx_pool = idx_all[mask]  # Pool of potential controls
            scores = (match[idx_pool] - match[idx]) ** 2  # Compute squared differences
            scores = np.sum(scores, axis=1)  # Sum across features to get distance

            # Shuffle to handle ties randomly
            shuffled = random_state.permutation(np.column_stack((scores, idx_pool)))
            t_idx = np.nanargmin(shuffled.T[0])  # Index of minimum distance

            score_match = shuffled.T[0][t_idx]  # Distance for diagnostics
            matching_scores.append(score_match)

            idx_match = shuffled.T[1][t_idx].astype(int)  # Matched control index
            control_group.append(idx_match)
            mask[idx_match] = False  # Remove matched control from pool

            assert mask_orig[idx_match], "Matched index was not originally masked."

        # Combine patient and control indices
        split[idx_set] = np.hstack((split[idx_set], control_group))

    # Update split dictionary with average matching score and adjusted sample size
    split = {
        "idx_train": split["idx_train"].tolist(),  # Convert to list
        "idx_val": split["idx_val"].tolist(),      # Convert to list
        "idx_test": split["idx_test"].tolist(),    # Convert to list
        "samplesize": split["samplesize"],
        "seed": split["seed"],
        "stratify": split["stratify"],
        "average_matching_score": np.mean(matching_scores),
    }
    split["samplesize"] *= 2  # Account for both patient and control groups

    return split


def write_splitfile(
    features_path: str,
    targets_path: str,
    split_path: str,
    confounds_path: str,
    confound_correction_method: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    stratify: bool = False,
    balanced: bool = False,
):
    """
    Generate a split file for a given dataset.

    Args:
        features_path (str): Path to the features HDF5 file.
        targets_path (str): Path to the targets HDF5 file.
        split_path (str): Path to save the generated split file.
        confounds_path (str): Path to the confounds HDF5 file.
        confound_correction_method (str): Method for confound correction.
        n_train (int): Number of training samples.
        n_val (int): Number of validation samples.
        n_test (int): Number of test samples.
        seed (int): Random seed for reproducibility.
        stratify (bool, optional): Whether to use stratified splitting. Defaults to False.
        balanced (bool, optional): Whether to balance classes. Defaults to False.
    """
    # Validate confound correction method
    valid_methods = ["correct-x", "correct-y", "correct-both", "none", "matching"]
    if confound_correction_method not in valid_methods:
        raise ValueError(f"Invalid confound correction method: {confound_correction_method}. "
                         f"Valid methods are: {', '.join(valid_methods)}")

    # Load masks and data from HDF5 files
    with h5py.File(features_path, "r") as f:
        x_mask = f["mask"][:]

    with h5py.File(targets_path, "r") as f:
        y = f["data"][:]
        y_mask = f["mask"][:]

    with h5py.File(confounds_path, "r") as f:
        confounds = f["data"][:]
        confounds_mask = f["mask"][:]

    # Combine masks to identify valid samples
    xy_mask = np.logical_and(x_mask, y_mask)

    # Check number of unique classes
    n_classes = len(np.unique(y[xy_mask]))
    if n_classes <= 1:
        # Insufficient classes for splitting
        with open(split_path, "w") as f:
            json.dump({"error": "insufficient samples"}, f, cls=NpEncoder, indent=0)
        return

    # Determine if stratification is feasible
    stratify = bool(stratify and n_classes <= 10)

    if confound_correction_method == "matching":
        # Perform matching-based split for binary classification
        xyc_mask = np.logical_and(xy_mask, confounds_mask)
        assert n_classes == 2, "Matching requires binary classification with class '1' as positive."

        required_samples = n_train // 2 + n_val // 2 + n_test // 2
        if sum(xyc_mask[y == 1]) > required_samples:
            split_dict = generate_matched_split(
                y=y,
                match=confounds,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                do_stratify=True,
                mask=xyc_mask,
                seed=seed,
            )
        else:
            # Not enough positive samples for matching
            split_dict = {"error": "insufficient samples"}
    else:
        if balanced:
            # Apply random undersampling to balance classes
            try:
                idx_all = np.arange(len(y))
                idx_undersampled, _ = RandomUnderSampler(random_state=seed).fit_resample(
                    idx_all[xy_mask].reshape(-1, 1), y[xy_mask].astype(int)
                )
                idx_undersampled = idx_undersampled.reshape(-1)
                xy_mask[[i for i in idx_all if i not in idx_undersampled]] = False
            except ValueError as e:
                # Handle cases with too few samples after undersampling
                error_message = f"""
                Undersampling failed: {str(e)}
                Possible reasons:
                - Too few samples in some classes.
                - Continuous labels may have been used unintentionally.
                Note: There are {n_classes} unique classes and {len(y[xy_mask])} samples in the target file.
                """
                raise ValueError(error_message) from e

        # Check if sufficient samples are available for splitting
        if sum(xy_mask) >= n_train + n_val + n_test:
            split_dict = generate_random_split(
                y=y,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                do_stratify=stratify,
                mask=xy_mask,
                seed=seed,
            )
        else:
            # Not enough samples for the desired split
            split_dict = {"error": "insufficient samples"}

    # Ensure indices are sorted for HDF5 compatibility
    if 'error' not in split_dict:
        for set_name in ["idx_train", "idx_val", "idx_test"]:
            split_dict[set_name] = sorted(split_dict[set_name])

    # Write the split information to a JSON file
    with open(split_path, "w") as f:
        json.dump(split_dict, f, cls=NpEncoder, indent=0)


if __name__ == "__main__":
    # Parse parameters from Snakemake wildcards and params
    n_train = int(snakemake.wildcards.samplesize)
    val_test_frac = snakemake.params.val_test_frac
    val_test_max = snakemake.params.val_test_max
    val_test_min = snakemake.params.val_test_min

    # Calculate number of validation and test samples with constraints
    n_val = n_test = min(
        round(n_train * val_test_frac), val_test_max
    ) if val_test_max else round(n_train * val_test_frac)
    n_val = n_test = max(n_val, val_test_min) if val_test_min else n_val

    # Ensure sample sizes are valid
    assert n_train > 1 and n_val > 1 and n_test > 1, "Sample sizes must be greater than 1."

    # Generate and write the split file
    write_splitfile(
        features_path=snakemake.input.features,
        targets_path=snakemake.input.targets,
        split_path=snakemake.output.split,
        confounds_path=snakemake.input.cni,
        confound_correction_method=snakemake.wildcards.confound_correction_method,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=int(snakemake.wildcards.seed),
        stratify=snakemake.params.stratify,
        balanced=True if snakemake.wildcards.balanced == 'True' else False,
    )