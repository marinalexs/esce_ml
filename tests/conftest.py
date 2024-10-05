"""
conftest.py
===========

This module contains pytest fixtures for the ESCE workflow tests.
These fixtures provide reusable test data and utilities across multiple test files.
"""

import pytest
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Callable, Tuple, Dict, List, Union, Any
import re

@pytest.fixture
def generate_synth_data() -> Callable[[int, int, bool, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fixture to generate synthetic data for testing.

    Returns:
        Callable: Function to generate synthetic data (X, y, confounds).
    """
    def _generate_synth_data(n_samples: int = 100, n_features: int = 10, classification: bool = True, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(random_state)
        X = np.random.randn(n_samples, n_features)
        confounds = np.random.randn(n_samples, 3)  # 3 confounding variables
        
        if classification:
            y = np.random.randint(0, 2, n_samples)
        else:
            y = np.random.randn(n_samples)
        
        return X, y, confounds
    
    return _generate_synth_data

@pytest.fixture
def write_data_to_file() -> Callable[[np.ndarray, str, Path], Path]:
    """
    Fixture to write data to a file in various formats.

    Returns:
        Callable: Function to write data to a file.
    """
    def _write_data_to_file(data: np.ndarray, file_format: str, file_path: Path) -> Path:
        if file_format == 'csv':
            pd.DataFrame(data).to_csv(file_path, index=False)
        elif file_format == 'tsv':
            pd.DataFrame(data).to_csv(file_path, sep='\t', index=False)
        elif file_format == 'npy':
            np.save(file_path, data)
        elif file_format == 'h5':
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('data', data=data)
                mask = ~np.isnan(data)
                if data.ndim == 2:
                    mask = np.all(mask, axis=1)
                f.create_dataset('mask', data=mask)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return file_path
    
    return _write_data_to_file

@pytest.fixture
def create_dataset(write_data_to_file: Callable) -> Callable[[np.ndarray, np.ndarray, np.ndarray, str, Path], Dict[str, Path]]:
    """
    Fixture to create a dataset from X, y, and confounds data.

    Args:
        write_data_to_file: Fixture to write data to a file.

    Returns:
        Callable: Function to create a dataset and return file paths.
    """
    def _create_dataset(X: np.ndarray, y: np.ndarray, confounds: np.ndarray, file_format: str, base_path: Path) -> Dict[str, Path]:
        features_path = write_data_to_file(X, file_format, base_path.with_name(f"{base_path.name}_features.{file_format}"))
        targets_path = write_data_to_file(y, file_format, base_path.with_name(f"{base_path.name}_targets.{file_format}"))
        confounds_path = write_data_to_file(confounds, file_format, base_path.with_name(f"{base_path.name}_confounds.{file_format}"))
        
        return {
            'features': features_path,
            'targets': targets_path,
            'confounds': confounds_path
        }
    
    return _create_dataset

@pytest.fixture
def generate_stats_data() -> Callable[[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fixture to generate synthetic stats data.

    Returns:
        Callable: Function to generate synthetic stats data.
    """
    def _generate_stats_data(path: str, random_state: int = 42, n_bootstrap: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(random_state)
        x = np.linspace(0, 1, 100)
        
        # Generate parameters for power law
        p0 = np.random.uniform(0, 0.5)
        p1 = np.random.uniform(0.5, 1.5)
        p2 = np.random.uniform(1, 3)
        
        y = p0 + p1 * x**p2
        y_err = np.random.normal(0, 0.1, size=y.shape)
        
        # Generate bootstrap samples
        bootstrap_params = np.random.normal(loc=[p0, p1, p2], 
                                            scale=[0.1, 0.1, 0.1], 
                                            size=(n_bootstrap, 3))
        
        return x, y, y_err, bootstrap_params
    
    return _generate_stats_data

@pytest.fixture
def write_stats_data() -> Callable[[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[str, str]]:
    """
    Fixture to write stats data to JSON files.

    Returns:
        Callable: Function to write stats data to JSON files.
    """
    def _write_stats_data(path: str, x: np.ndarray, y: np.ndarray, y_err: np.ndarray, bootstrap_params: np.ndarray) -> Tuple[str, str]:
        stats_file = f"{path}_stats.json"
        bootstrap_file = f"{path}_bootstrap.json"
        
        stats_data = {
            'x': x.tolist(),
            'y_mean': y.tolist(),
            'y_std': y_err.tolist()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f)
        
        with open(bootstrap_file, 'w') as f:
            json.dump(bootstrap_params.tolist(), f)
        
        return stats_file, bootstrap_file
    
    return _write_stats_data

@pytest.fixture
def parse_filename() -> Callable[[str], Dict[str, str]]:
    """
    Fixture to parse a filename and extract variables.

    Returns:
        Callable: Function to parse a filename and return a dictionary of variables.
    """
    def _parse_filename(filename: str) -> Dict[str, str]:
        pattern = r"(?P<features>[\w-]+)_(?P<targets>[\w-]+)_(?P<confound_correction_method>[\w-]+)_(?P<confound_correction_cni>[\w-]+)_(?P<balanced>[\w-]+)_(?P<grid>[\w-]+)"
        match = re.search(pattern, filename)
        
        if match:
            return match.groupdict()
        else:
            raise ValueError(f"Unable to parse filename: {filename}")
    
    return _parse_filename

