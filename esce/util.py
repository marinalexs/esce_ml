"""This module provides utility functions required by other modules."""

import gzip
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
import requests
import yaml
from tqdm import tqdm


def dropna(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop infinity and NaNs from both arrays aligned.

    Arguments:
        x: First array to check
        y: Second array to check

    Returns:
        Tuple (x,y) where both do not contain infinity/NaN values
    """
    mask_x = np.isfinite(x).all(1)
    mask_y = np.isfinite(y)
    mask = np.logical_and(mask_x, mask_y)
    return x[mask], y[mask]


def flip(x: np.ndarray, prob: float, seed: int) -> np.ndarray:
    """Randomly flip an array of binary labels.

    Arguments:
        x: Array of binary data
        prob: Probability to flip a label
        seed: Seed to label random engine by

    Returns:
        Array of new labels
    """
    np.random.seed(seed)
    indices = np.random.random(x.shape) < prob
    return np.logical_xor(x, indices).astype(int)


def flt2str(f: float, decimals: int = 4) -> str:
    """Serialize a float to a string.

    Arguments:
        f: Floating point value to convert
        decimals: How many decimal places to store

    Returns:
        Rounded float with full stops replaced by "#"
    """
    return str(round(f, decimals)).replace(".", "#")


def hash_dict(x: Dict[Any, Any]) -> str:
    """Compute the hash of a dictionary.

    Sorts the keys, converts it to JSON and computes an MD5 hash.

    Arguments:
        x: Dictionary to compute hash of

    Returns:
        Hexdigest string representation of the hashed dictionary
    """
    return hashlib.md5(json.dumps(x, sort_keys=True).encode("utf-8")).hexdigest()


def load_grid_file(grid_name: str) -> Any:
    """Load a grid from a YAML file.

    Grid YAML files may contain multiple grids.
    These grid can be selected by adding @<entry> to
    the file path, e.g. grid.yaml@grid1

    Arguments:
        grid_name: Path to the YAML file.

    Returns:
        Grid dictionary
    """
    grid_key = None
    if "@" in grid_name:
        grid_path, grid_key = grid_name.split("@")
    else:
        grid_path = grid_name

    grid_file = Path(grid_path)
    if grid_file.is_file():
        with grid_file.open("r") as f:
            grid = yaml.safe_load(f)
            if grid_key is not None:
                grid = grid[grid_key]
            return grid
    else:
        raise ValueError("Invalid grid file path")


def load_dataset(
    data_path: Path, label: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Load data and label from data file.

    Arguments:
        data_path: Path to h5 or pkl file
        label: Label to use contained in the file

    Returns:
        (x,y) Tuple of data and labels
    """

    if data_path.suffix == ".h5":
        with h5py.File(data_path, "r") as f:
            x = f["/data"][...]
            if label is None:
                return x
            y = f[f"/labels/{label}"][...]
    elif data_path.suffix == ".pkl":
        with data_path.open("rb") as f:
            d = pickle.load(f)
            x = d["data"]
            if label is None:
                return x
            y = d[f"label_{label}"]
    else:
        raise ValueError("Unknown file format")

    return dropna(x, y)


def load_split(split_path: Path) -> Any:
    """Load a split file from a path.

    Arguments:
        split_path: Path to split file

    Returns:
        Tuple consisting of a seed and the splits
    """

    with split_path.open("rb") as f:
        return pickle.load(f)


def download_file(url: str, path: Path) -> None:
    """Download the URL and save it in the given path.

    Arguments:
        url: URL to download
        path: Path to store file to
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    bar = tqdm(total=total, unit="iB", unit_scale=True)
    block_size = 1024
    with path.open("wb") as f:
        for data in resp.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
    bar.close()
    if total != 0 and bar.n != total:
        raise ValueError


def extract_gzip(in_path: Path, out_path: Path) -> None:
    """Extract a GZip file to a given path.

    Arguments:
        in_path: Path to the GZip file
        out_path: Where to write the unzipped contents to
    """
    with open(out_path, "wb") as f_out:
        with gzip.open(in_path, "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)
