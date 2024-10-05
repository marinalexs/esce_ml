"""
This module provides functions for extrapolating learning curves and performing bootstrap analysis.
It includes utilities for fitting power law models to data and handling JSON serialization of NumPy types.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import r2_score

MIN_DOF = 2  # Minimum degrees of freedom required for curve fitting

# Set up logging
log_level = os.environ.get('ESCE_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""

    def default(self, obj: Union[np.integer, np.floating, np.ndarray]) -> Union[int, float, List]:
        """
        Convert NumPy data types to native Python types for JSON serialization.

        Args:
            obj: The object to be encoded.

        Returns:
            The encoded object as a native Python type.

        Raises:
            TypeError: If the object type is not supported.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def power_law_model(t: Union[float, np.ndarray], a: float, b: float, c: float) -> Union[float, np.ndarray]:
    """
    Power law model function: y = a * x^(-b) + c

    Args:
        t: Independent variable(s).
        a: Amplitude parameter.
        b: Decay rate parameter.
        c: Offset parameter.

    Returns:
        Predicted value(s) based on the power law model.
    """
    return a * t ** (-b) + c


def fit_curve(x: np.ndarray, y: np.ndarray, y_e: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """
    Fit a power law curve to the data.

    Args:
        x: Independent variable data.
        y: Dependent variable data (mean scores).
        y_e: Standard error of y.

    Returns:
        Dictionary containing fitted parameters and goodness-of-fit metrics.
    """
    result: Dict[str, Union[np.ndarray, float]] = {
        "p_mean": np.array([np.nan, np.nan, np.nan]),
        "r2": np.nan,
        "chi2": np.nan,
        "mu": np.nan,
        "sigma": np.nan,
    }

    dof = len(x) - 3  # Degrees of freedom: number of data points minus number of parameters
    if dof < MIN_DOF:
        logging.warning(f"Insufficient degrees of freedom (dof = {dof} < {MIN_DOF})")
        return result

    try:
        # Fit the power law model using non-linear least squares
        p_mean, _ = scipy.optimize.curve_fit(
            power_law_model,
            x,
            y,
            # sigma=y_e,  # Uncomment if y_e represents standard errors
            maxfev=5000,  # Maximum number of function evaluations
            p0=(-1, 0.1, 0.5),  # Initial guess for parameters
            bounds=((-np.inf, 0, -np.inf), (0, np.inf, np.inf)),  # Parameter bounds
        )
        result["p_mean"] = p_mean

        # Calculate R² score for goodness of fit
        y_pred = power_law_model(x, *p_mean)
        result["r2"] = r2_score(y, y_pred)

        # Calculate chi-squared statistic
        result["chi2"] = np.sum(((y - y_pred) / y_e) ** 2) / dof

        # Calculate mean and standard deviation of residuals normalized by standard error
        residuals = (y - y_pred) / y_e
        result["mu"], result["sigma"] = np.mean(residuals), np.std(residuals)

        logging.info(f"Curve fitting successful. R² = {result['r2']:.4f}, Chi² = {result['chi2']:.4f}")
        return result
    except Exception as e:
        logging.error(f"Curve fitting failed: {e}")
        return result


def extrapolate(
    stats_path: str,
    extra_path: str,
    bootstrap_path: str,
    repeats: int,
) -> None:
    """
    Fit a power law model to the scores and perform bootstrap analysis for uncertainties.

    Args:
        stats_path: Path to the input stats CSV file.
        extra_path: Path to save the extrapolation results as a JSON file.
        bootstrap_path: Path to save the bootstrap results as a JSON file.
        repeats: Number of bootstrap repetitions.
    """
    logging.info(f"Starting extrapolation process with {repeats} bootstrap repetitions")

    # Check if the stats file is empty
    if os.stat(stats_path).st_size == 0:
        logging.warning(f"Input stats file {stats_path} is empty. Creating empty output files.")
        Path(extra_path).touch()
        Path(bootstrap_path).touch()
        return

    # Load the scores into a DataFrame
    df = pd.read_csv(stats_path, index_col=False)
    logging.info(f"Loaded data from {stats_path}. Shape: {df.shape}")

    # Determine the metric to use based on available columns
    metric = "r2_test" if "r2_test" in df.columns else "acc_test"
    logging.info(f"Using metric: {metric}")

    result: Dict[str, Union[int, str, List[float], List[bool]]] = {"n_seeds": len(df["s"].unique())}

    # Aggregate mean, std, and SEM for each sample size
    grouped = df.groupby("n")[metric]
    x = grouped.mean().index.values
    y_mean = grouped.mean().values
    y_std = grouped.std().values
    y_sem = grouped.sem().values
    mask = (y_mean - y_sem) > 0

    # Update the result dictionary with aggregated data
    result.update(
        {
            "metric": metric,
            "x": x.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
            "y_sem": y_sem.tolist(),
            "mask": mask.tolist(),
            "dof": int(sum(mask) - 3),
        }
    )

    logging.info(f"Data aggregated. Number of sample sizes: {len(x)}")

    # Fit the power law curve to the masked data
    fit_result = fit_curve(x[mask], y_mean[mask], y_sem[mask])
    result.update(fit_result)

    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Modify the bootstrap section
    if result["dof"] < MIN_DOF:
        logging.warning(f"Insufficient degrees of freedom (dof = {result['dof']} < {MIN_DOF}). Skipping bootstrap.")
        # Set bootstrap results to NaN and create a bootstrap file with an empty list
        result.update({
            "p_bootstrap_mean": [np.nan] * 3,
            "p_bootstrap_std": [np.nan] * 3,
            "p_bootstrap_975": [np.nan] * 3,
            "p_bootstrap_025": [np.nan] * 3,
            "y_bootstrap_mean": [np.nan] * len(x),
            "y_bootstrap_std": [np.nan] * len(x),
            "y_bootstrap_975": [np.nan] * len(x),
            "y_bootstrap_025": [np.nan] * len(x),
        })
        # Create a bootstrap file with an empty list
        with open(bootstrap_path, "w") as f:
            json.dump([], f)
    else:
        # Perform bootstrap repetitions
        logging.info(f"Starting bootstrap analysis with {repeats} repetitions")
        p_bootstrap: List[List[float]] = []
        y_bootstrap: List[List[float]] = [[] for _ in x]

        for i in range(repeats):
            y_bs_sample = df.groupby("n")[metric].apply(lambda g: g.sample(n=len(g), replace=True))
            y_bs_sample_mean = y_bs_sample.groupby("n").mean()
            y_bs_sample_sem = y_bs_sample.groupby("n").sem()

            for j, n in enumerate(x):
                y_bootstrap[j].append(y_bs_sample_mean[n])

            p_ = fit_curve(x[mask], y_bs_sample_mean[mask], y_bs_sample_sem[mask])["p_mean"]
            if np.isfinite(p_).all():
                p_bootstrap.append(p_.tolist())
            
            if (i + 1) % 100 == 0:
                logging.info(f"Completed {i + 1} bootstrap repetitions")

        logging.info(f"Bootstrap completed: {len(p_bootstrap)} successful out of {repeats} repetitions")

        # Calculate and store bootstrap statistics
        if len(p_bootstrap) > 0.9 * repeats:
            p_bootstrap_array = np.array(p_bootstrap)
            result.update({
                "p_bootstrap_mean": np.mean(p_bootstrap_array, axis=0).tolist(),
                "p_bootstrap_std": np.std(p_bootstrap_array, axis=0).tolist(),
                "p_bootstrap_975": np.percentile(p_bootstrap_array, 97.5, axis=0).tolist(),
                "p_bootstrap_025": np.percentile(p_bootstrap_array, 2.5, axis=0).tolist(),
            })
            logging.info("Bootstrap statistics calculated successfully")
        else:
            logging.warning("More than 10% of bootstrap repetitions failed. Setting bootstrap results to NaN.")
            result.update({
                "p_bootstrap_mean": [np.nan] * 3,
                "p_bootstrap_std": [np.nan] * 3,
                "p_bootstrap_975": [np.nan] * 3,
                "p_bootstrap_025": [np.nan] * 3,
            })

        # Add bootstrap mean and standard deviation to the result
        y_bootstrap_array = np.array(y_bootstrap)
        result.update({
            "y_bootstrap_mean": np.mean(y_bootstrap_array, axis=1).tolist(),
            "y_bootstrap_std": np.std(y_bootstrap_array, axis=1).tolist(),
            "y_bootstrap_975": np.percentile(y_bootstrap_array, 97.5, axis=1).tolist(),
            "y_bootstrap_025": np.percentile(y_bootstrap_array, 2.5, axis=1).tolist(),
        })

        # Save the bootstrap results to the specified JSON file
        with open(bootstrap_path, "w") as f:
            json.dump(p_bootstrap, f, cls=NpEncoder, indent=0)

    # Save the extrapolation results to the specified JSON file
    with open(extra_path, "w") as f:
        json.dump(result, f, cls=NpEncoder, indent=0)
    logging.info(f"Extrapolation results saved to {extra_path}")

    logging.info("Extrapolation process completed successfully")


if __name__ == "__main__":
    extrapolate(
        stats_path=snakemake.input.scores,
        extra_path=snakemake.output.stats,
        bootstrap_path=snakemake.output.bootstraps,
        repeats=snakemake.params.bootstrap_repetitions,
    )