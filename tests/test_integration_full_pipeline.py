import tempfile
import os
import json
import h5py
import pandas as pd
import numpy as np
import pytest
import shutil
from datetime import datetime

# Import the necessary functions
from workflow.scripts.prepare_data import prepare_data
from workflow.scripts.fit_model import fit
from workflow.scripts.aggregate import aggregate
from workflow.scripts.extrapolate import extrapolate
from workflow.scripts.plot import plot
from workflow.scripts.plot_hps import plot as plot_hps

# Add this function at the beginning of the file
def create_debug_tmpdir():
    """
    Create a temporary directory for debugging within the tests folder.
    
    Returns:
        str: Path to the created temporary directory.
    """
    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_tmp')
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = os.path.join(debug_dir, f'test_run_{timestamp}')
    os.makedirs(tmp_dir, exist_ok=True)  # Add exist_ok=True to prevent FileExistsError
    return tmp_dir

def create_synthetic_data(tmpdir, n_samples=1000, n_features=10, binary=True):
    """
    Generate synthetic datasets for integration testing.

    Args:
        tmpdir (str): Temporary directory path.
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        binary (bool): Whether to generate binary targets.

    Returns:
        Tuple[str, str, str]: Paths to features, targets, and covariates HDF5 files.
    """
    features = pd.DataFrame(np.random.rand(n_samples, n_features), 
                            columns=[f'feature_{i}' for i in range(n_features)])
    targets = pd.DataFrame({'target': np.random.randint(0, 2, n_samples) if binary else np.random.rand(n_samples)})
    covariates = pd.DataFrame(np.random.rand(n_samples, 3), columns=['cov1', 'cov2', 'cov3'])

    features_path = os.path.join(tmpdir, 'features.csv')
    targets_path = os.path.join(tmpdir, 'targets.csv')
    covariates_path = os.path.join(tmpdir, 'covariates.csv')

    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)
    covariates.to_csv(covariates_path, index=False)

    return features_path, targets_path, covariates_path

def test_full_pipeline_classification():
    """
    Integration test for the full pipeline using a classification model with multiple sample sizes.
    
    This test verifies that data preparation, model fitting, evaluation,
    aggregation, extrapolation, plotting, and hyperparameter visualization work together without issues
    for various sample sizes.
    """
    tmpdir = create_debug_tmpdir()
    try:
        # Step 1: Generate synthetic data
        features_csv, targets_csv, covariates_csv = create_synthetic_data(tmpdir, n_samples=1000, binary=True)
        
        # Step 2: Convert CSVs to HDF5
        features_h5 = os.path.join(tmpdir, 'features.h5')
        targets_h5 = os.path.join(tmpdir, 'targets.h5')
        covariates_h5 = os.path.join(tmpdir, 'covariates.h5')

        prepare_data(features_h5, 'pytest_dataset', 'features', 'normal', 
                    {'pytest_dataset': {'features': {'normal': features_csv}}})
        prepare_data(targets_h5, 'pytest_dataset', 'targets', 'normal', 
                    {'pytest_dataset': {'targets': {'normal': targets_csv}}})
        prepare_data(covariates_h5, 'pytest_dataset', 'covariates', 'normal', 
                    {'pytest_dataset': {'covariates': {'normal': covariates_csv}}})

        # Step 3: Generate data splits for multiple sample sizes
        sample_sizes = [100, 300, 600, 1000]
        split_paths = []
        
        for size in sample_sizes:
            split_path = os.path.join(tmpdir, f'split_{size}.json')
            with open(split_path, 'w') as f:
                split = {
                    "idx_train": list(range(int(size * 0.6))),
                    "idx_val": list(range(int(size * 0.6), int(size * 0.8))),
                    "idx_test": list(range(int(size * 0.8), size)),
                    "samplesize": size,
                    "seed": 42
                }
                json.dump(split, f)
            split_paths.append(split_path)

        # Step 4: Define model and hyperparameters
        model_name = "ridge-cls"
        grid = {"ridge-cls": {"alpha": [0.1, 1.0, 10.0]}}

        # Step 5: Run the fit function for each sample size
        scores_paths = []
        for split_path in split_paths:
            scores_path = os.path.join(tmpdir, f'scores_{os.path.basename(split_path)}.csv')
            fit(
                features_path=features_h5,
                targets_path=targets_h5,
                split_path=split_path,
                scores_path=scores_path,
                model_name=model_name,
                grid=grid,
                existing_scores_path_list=[],
                confound_correction_method='normal',
                cni_path=covariates_h5
            )
            scores_paths.append(scores_path)

        # Step 6: Aggregate results
        aggregated_path = os.path.join(tmpdir, 'aggregated.csv')
        aggregate(scores_paths, aggregated_path)

        # Step 7: Extrapolate results
        extra_path = os.path.join(tmpdir, 'stats.json')
        bootstrap_path = os.path.join(tmpdir, 'bootstrap.json')
        extrapolate(aggregated_path, extra_path, bootstrap_path, repeats=10)

        # Step 8: Plot results
        plot_path = os.path.join(tmpdir, 'plot.html')
        plot([extra_path], plot_path, color_variable='dataset', linestyle_variable=None, title='Integration Test Plot')

        # Step 9: Plot hyperparameters
        hp_plot_path = os.path.join(tmpdir, 'hp_plot.html')
        plot_hps(
            stats_filename=aggregated_path,
            output_filename=hp_plot_path,
            grid=grid,
            hyperparameter_scales={"alpha": "log"},
            model_name=model_name,
            title="Hyperparameter Plot - Classification"
        )

        # Assertions
        for scores_path in scores_paths:
            assert os.path.exists(scores_path), f"Scores file {scores_path} was not created."
        assert os.path.exists(aggregated_path), "Aggregated file was not created."
        assert os.path.exists(extra_path), "Extrapolation file was not created."
        assert os.path.exists(bootstrap_path), "Bootstrap file was not created."
        assert os.path.exists(plot_path), "Plot file was not created."
        
        # Further assertions
        for scores_path in scores_paths:
            scores_df = pd.read_csv(scores_path)
            assert not scores_df.empty, f"Scores DataFrame for {scores_path} is empty."

        aggregated_df = pd.read_csv(aggregated_path)
        print("Columns in aggregated_df:", aggregated_df.columns)
        print("First few rows of aggregated_df:")
        print(aggregated_df.head())
        
        assert not aggregated_df.empty, "Aggregated DataFrame is empty."
        assert len(aggregated_df['n'].unique()) == len(sample_sizes), "Not all sample sizes are present in aggregated results."

        with open(extra_path, 'r') as f:
            extra_data = json.load(f)
            assert "metric" in extra_data, "Extrapolate output missing 'metric'."

        with open(bootstrap_path, 'r') as f:
            bootstrap_data = json.load(f)
            assert isinstance(bootstrap_data, list), "Bootstrap data should be a list."

        assert os.path.getsize(plot_path) > 0, "Plot file is empty."
        assert os.path.exists(hp_plot_path), "Hyperparameter plot file was not created."
        assert os.path.getsize(hp_plot_path) > 0, "Hyperparameter plot file is empty."

    finally:
        print(f"Debug files saved in: {tmpdir}")

def test_full_pipeline_regression():
    """
    Integration test for the full pipeline using a regression model with multiple sample sizes.
    
    This test verifies that data preparation, model fitting, evaluation,
    aggregation, extrapolation, plotting, and hyperparameter visualization work together seamlessly for regression tasks
    with various sample sizes.
    """
    tmpdir = create_debug_tmpdir()
    try:
        # Step 1: Generate synthetic data
        features_csv, targets_csv, covariates_csv = create_synthetic_data(tmpdir, n_samples=1000, binary=False)
        
        # Step 2: Convert CSVs to HDF5
        features_h5 = os.path.join(tmpdir, 'features.h5')
        targets_h5 = os.path.join(tmpdir, 'targets.h5')
        covariates_h5 = os.path.join(tmpdir, 'covariates.h5')

        prepare_data(features_h5, 'pytest_dataset', 'features', 'normal', 
                    {'pytest_dataset': {'features': {'normal': features_csv}}})
        prepare_data(targets_h5, 'pytest_dataset', 'targets', 'normal', 
                    {'pytest_dataset': {'targets': {'normal': targets_csv}}})
        prepare_data(covariates_h5, 'pytest_dataset', 'covariates', 'normal', 
                    {'pytest_dataset': {'covariates': {'normal': covariates_csv}}})

        # Step 3: Generate data splits for multiple sample sizes
        sample_sizes = [100, 300, 600, 1000]
        split_paths = []
        
        for size in sample_sizes:
            split_path = os.path.join(tmpdir, f'split_{size}.json')
            with open(split_path, 'w') as f:
                split = {
                    "idx_train": list(range(int(size * 0.6))),
                    "idx_val": list(range(int(size * 0.6), int(size * 0.8))),
                    "idx_test": list(range(int(size * 0.8), size)),
                    "samplesize": size,
                    "seed": 42
                }
                json.dump(split, f)
            split_paths.append(split_path)

        # Step 4: Define model and hyperparameters
        model_name = "ridge-reg"
        grid = {"ridge-reg": {"alpha": [0.1, 1.0, 10.0]}}

        # Step 5: Run the fit function for each sample size
        scores_paths = []
        for split_path in split_paths:
            scores_path = os.path.join(tmpdir, f'scores_{os.path.basename(split_path)}.csv')
            fit(
                features_path=features_h5,
                targets_path=targets_h5,
                split_path=split_path,
                scores_path=scores_path,
                model_name=model_name,
                grid=grid,
                existing_scores_path_list=[],
                confound_correction_method='normal',
                cni_path=covariates_h5
            )
            scores_paths.append(scores_path)

        # Step 6: Aggregate results
        aggregated_path = os.path.join(tmpdir, 'aggregated.csv')
        aggregate(scores_paths, aggregated_path)

        # Step 7: Extrapolate results
        extra_path = os.path.join(tmpdir, 'stats.json')
        bootstrap_path = os.path.join(tmpdir, 'bootstrap.json')
        extrapolate(aggregated_path, extra_path, bootstrap_path, repeats=10)

        # Step 8: Plot results
        plot_path = os.path.join(tmpdir, 'plot.html')
        plot([extra_path], plot_path, color_variable='dataset', linestyle_variable=None, title='Integration Test Plot')

        # Step 9: Plot hyperparameters
        hp_plot_path = os.path.join(tmpdir, 'hp_plot.html')
        plot_hps(
            stats_filename=aggregated_path,
            output_filename=hp_plot_path,
            grid=grid,
            hyperparameter_scales={"alpha": "log"},
            model_name=model_name,
            title="Hyperparameter Plot - Regression"
        )

        # Assertions
        for scores_path in scores_paths:
            assert os.path.exists(scores_path), f"Scores file {scores_path} was not created."
        assert os.path.exists(aggregated_path), "Aggregated file was not created."
        assert os.path.exists(extra_path), "Extrapolation file was not created."
        assert os.path.exists(bootstrap_path), "Bootstrap file was not created."
        assert os.path.exists(plot_path), "Plot file was not created."
        
        # Further assertions
        for scores_path in scores_paths:
            scores_df = pd.read_csv(scores_path)
            assert not scores_df.empty, f"Scores DataFrame for {scores_path} is empty."

        aggregated_df = pd.read_csv(aggregated_path)
        print("Columns in aggregated_df:", aggregated_df.columns)
        print("First few rows of aggregated_df:")
        print(aggregated_df.head())
        
        assert not aggregated_df.empty, "Aggregated DataFrame is empty."
        assert len(aggregated_df['n'].unique()) == len(sample_sizes), "Not all sample sizes are present in aggregated results."

        with open(extra_path, 'r') as f:
            extra_data = json.load(f)
            assert "metric" in extra_data, "Extrapolate output missing 'metric'."

        with open(bootstrap_path, 'r') as f:
            bootstrap_data = json.load(f)
            assert isinstance(bootstrap_data, list), "Bootstrap data should be a list."

        assert os.path.getsize(plot_path) > 0, "Plot file is empty."
        assert os.path.exists(hp_plot_path), "Hyperparameter plot file was not created."
        assert os.path.getsize(hp_plot_path) > 0, "Hyperparameter plot file is empty."

    finally:
        print(f"Debug files saved in: {tmpdir}")