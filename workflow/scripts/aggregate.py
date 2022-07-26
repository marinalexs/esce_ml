import os
from pathlib import Path

import pandas as pd


def aggregate(
    score_path_list: str,
    stats_path: str,
) -> None:

    df_list = []
    for filename in score_path_list:
        if os.stat(filename).st_size > 0:
            df_list.append(pd.read_csv(filename, index_col=False))

    if not df_list:
        Path(stats_path).touch()
        return

    df = pd.concat(
        df_list,
        axis=0,
        ignore_index=True,
    )

    metric = "r2_val" if "r2_val" in df.columns else "acc_val"
    idx_best = df.groupby(["n", "s"])[metric].idxmax()
    df.loc[idx_best].to_csv(stats_path, index=False)


aggregate(snakemake.input.scores, snakemake.output.scores)
