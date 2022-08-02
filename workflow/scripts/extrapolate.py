import json

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import r2_score

MIN_DOF = 2


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def extrapolate(stats_path: str, extra_path: str, bootstrap_path: str, repeats: int):
    df = pd.read_csv(stats_path, index_col=False)
    metric = "r2_test" if "r2_test" in df.columns else "acc_test"
    result = {"n_seeds": len(df["s"].unique())}

    x, y_mean, y_std, y_sem, mask = [], [], [], [], []
    for n in sorted(df["n"].unique()):
        x.append(n)
        y_mean.append(df[df["n"] == n][metric].mean())
        y_std.append(df[df["n"] == n][metric].std())
        y_sem.append(df[df["n"] == n][metric].sem())
        mask.append(bool((y_mean[-1] - y_sem[-1]) > 0))
    x = np.asarray(x)
    y_mean = np.asarray(y_mean)
    y_std = np.asarray(y_std)
    y_sem = np.asarray(y_sem)

    result["metric"] = metric
    result["x"] = x
    result["y_mean"] = y_mean
    result["y_std"] = y_std
    result["y_sem"] = y_sem
    result["mask"] = mask
    result["dof"] = sum(mask) - 3

    if result["dof"] >= MIN_DOF:
        p_mean, _ = scipy.optimize.curve_fit(
            lambda t, a, b, c: a * t ** (-b) + c,
            x[mask],
            y_mean[mask],
            sigma=y_sem[mask],
            maxfev=5000,
            p0=(-1, 0.01, 0.7),
            bounds=((-np.inf, 0, 0), (0, 1, 1)),
        )

        result["p_mean"] = p_mean
        result["r2"] = r2_score(
            y_mean[mask], p_mean[0] * x[mask] ** (-p_mean[1]) + p_mean[2]
        )
        result["chi2"] = (
            sum(
                (y_mean[mask] - (p_mean[0] * x[mask] ** (-p_mean[1]) + p_mean[2])) ** 2
                / y_sem[mask] ** 2
            )
            / result["dof"]
        )
        result["mu"], result["sigma"] = np.mean(
            (y_mean[mask] - (p_mean[0] * x[mask] ** (-p_mean[1]) + p_mean[2]))
            / y_sem[mask]
        ), np.std(
            (y_mean[mask] - (p_mean[0] * x[mask] ** (-p_mean[1]) + p_mean[2]))
            / y_sem[mask]
        )
    else:
        (
            result["p_mean"],
            result["r2"],
            result["chi2"],
            result["mu"],
            result["sigma"],
        ) = ([np.nan, np.nan, np.nan], np.nan, np.nan, np.nan, np.nan)

    p_bootstrap = []
    y_bootstrap = [[] for _ in x]
    for _ in range(repeats):
        y_bs_sample_mean, y_bs_sample_std, y_bs_sample_sem = [], [], []
        for i, n in enumerate(x):
            y_bs_sample = df[df["n"] == n][metric].sample(frac=1, replace=True)
            y_bs_sample_mean.append(y_bs_sample.mean())
            y_bs_sample_sem.append(y_bs_sample.sem())
            y_bootstrap[i].append(y_bs_sample.mean())
        if result["dof"] >= MIN_DOF:
            try:  # for rare convergence issues
                p_, _ = scipy.optimize.curve_fit(
                    lambda t, a, b, c: a * t ** (-b) + c,
                    x[mask],
                    np.asarray(y_bs_sample_mean)[mask],
                    sigma=np.asarray(y_bs_sample_sem)[mask],
                    maxfev=5000,
                    p0=(-1, 0.01, 0.7),
                    bounds=((-np.inf, 0, 0), (0, 1, 1)),
                )
                p_bootstrap.append(p_)
            except:
                pass

    if result["dof"] >= MIN_DOF:
        result["p_bootstrap_mean"] = np.mean(p_bootstrap, 0)
        result["p_bootstrap_std"] = np.std(p_bootstrap, 0)
        result["p_bootstrap_975"] = np.percentile(p_bootstrap, 97.5, axis=0)
        result["p_bootstrap_025"] = np.percentile(p_bootstrap, 2.5, axis=0)
    else:
        result["p_bootstrap_mean"] = np.nan
        result["p_bootstrap_std"] = np.nan
        result["p_bootstrap_975"] = np.nan
        result["p_bootstrap_025"] = np.nan

    result["y_bootstrap_mean"] = np.mean(y_bootstrap, 1)
    result["y_bootstrap_std"] = np.std(y_bootstrap, 1)
    result["y_bootstrap_975"] = np.percentile(y_bootstrap, 97.5, axis=1)
    result["y_bootstrap_025"] = np.percentile(y_bootstrap, 2.5, axis=1)

    with open(extra_path, "w") as f:
        json.dump(result, f, cls=NpEncoder, indent=0)

    with open(bootstrap_path, "w") as f:
        json.dump(p_bootstrap, f, cls=NpEncoder, indent=0)


extrapolate(
    snakemake.input.scores,
    snakemake.output.stats,
    snakemake.output.bootstraps,
    snakemake.params.bootstrap_repetitions,
)
