import json

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import r2_score


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def extrapolate(
    stats_path: str, extra_path: str, bootstrap_path: str, repeats: int
) -> None:
    df = pd.read_csv(stats_path, index_col=False)
    metric = "r2_test" if "r2_test" in df.columns else "acc_test"
    result = {"n_seeds": len(df["s"].unique())}

    x, y_mean, y_std, y_sem = [], [], [], []
    for n in sorted(df["n"].unique()):
        x.append(n)
        y_mean.append(df[df["n"] == n][metric].mean())
        y_std.append(df[df["n"] == n][metric].std())
        y_sem.append(df[df["n"] == n][metric].std() / np.sqrt(result["n_seeds"]))

    p_mean, _ = scipy.optimize.curve_fit(
        lambda t, a, b, c: a * t ** (-b) + c,
        x,
        y_mean,
        sigma=None,
        maxfev=5000,
        p0=(-1, 0.01, 0.7),
        bounds=((-np.inf, 0, 0), (0, 1, 1)),
    )

    p_bootstrap = []
    y_bootstrap = [[] for _ in x]
    for _ in range(repeats):
        y = []
        for i, n in enumerate(x):
            y_sample = df[df["n"] == n][metric].sample(frac=1, replace=True).mean()
            y.append(y_sample)
            y_bootstrap[i].append(y_sample)
        try:
            p_, _ = scipy.optimize.curve_fit(
                lambda t, a, b, c: a * t ** (-b) + c,
                x,
                y,
                sigma=None,
                maxfev=5000,
                p0=(-1, 0.01, 0.7),
                bounds=((-np.inf, 0, 0), (0, 1, 1)),
            )
            p_bootstrap.append(p_)
        except RuntimeError:
            print("failed to fit")
            continue

    result["metric"] = metric
    result["x"] = x
    result["y_mean"] = y_mean
    result["y_std"] = y_std
    result["y_sem"] = y_sem

    x = np.asarray(x)
    y_mean = np.asarray(y_mean)
    y_sem = np.asarray(y_sem)

    result["p_mean"] = p_mean
    result["r2"] = r2_score(y_mean, p_mean[0] * x ** (-p_mean[1]) + p_mean[2])
    result["chi2"] = sum(
        (y_mean - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2])) ** 2 / y_sem**2
    ) / (len(x) - len(p_mean))
    result["mu"], result["sigma"] = np.mean(
        (y_mean - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2])) / y_sem
    ), np.std((y_mean - (p_mean[0] * x ** (-p_mean[1]) + p_mean[2])) / y_sem)

    result["p_bootstrap_mean"] = np.mean(p_bootstrap, 0)
    result["p_bootstrap_std"] = np.std(p_bootstrap, 0)
    result["p_bootstrap_975"] = np.percentile(p_bootstrap, 97.5, axis=0)
    result["p_bootstrap_025"] = np.percentile(p_bootstrap, 2.5, axis=0)

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
