import os
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import json
import re
import altair as alt

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
# this is a hack to make yaml load the floats correctly


def process_results(available_results):
    """Read available results and return them as a pandas dataframe."""
    df = pd.DataFrame(available_results, columns=["full_path"])
    df[
        [
            "dataset",
            "model",
            "features",
            "target",
            "confound_correction_method",
            "confound_correction_cni",
            "balanced",
            "grid",
        ]
    ] = (
        df["full_path"]
        .str.replace("results/", "")
        .str.replace("statistics/", "")
        .str.replace(".stats.json", "")
        .str.replace("/", "_")
        .str.split("_", expand=True)
    )
    df["cni"] = df[["confound_correction_method", "confound_correction_cni"]].agg("-".join, axis=1)
    return df


def plot(
    stats_file_list: str, output_filename: str, color_variable: Optional[str], linestyle_variable: Optional[str], title:str, max_x:int=6
):
    """Plot the results of a model using Altair.

    Args:
        stats_file_list: list of paths to the results files
        output_filename: path to save the plot
        color_variable: variable to use for color
        linestyle_variable: variable to use for linestyle
        title: title of the plot
        max_x: maximum sample size in powers of 10
    """
    print(stats_file_list)
    df = process_results(stats_file_list)

    data = []
    for _, row in df.iterrows():
        with open(row.full_path) as f:
            score = yaml.load(f, Loader=loader)
        if not score:
            print(row.full_path, "is empty - skipping")
            continue
            
        df_ = pd.DataFrame(
            {"n": score["x"], "y": score["y_mean"], "y_std": score["y_std"]}
        )
        df_["model"] = row.model
        df_["features"] = row.features
        df_["target"] = row.target
        df_["cni"] = row.cni
        data.append(df_)

    # skip if no data
    if len(data) > 0:
        data = pd.concat(data, axis=0, ignore_index=True)
        data = data.sort_values(color_variable)
    else:
        Path(output_filename).touch()
        return
    
    data['y'] = data['y'].clip(lower=-1)
    data['y_std'] = data['y_std'].clip(upper=1)
    print(data)

    # Create the main chart
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('n:Q', scale=alt.Scale(type='log')),
        y=alt.Y('y:Q', scale=alt.Scale(zero=True)),
        color=alt.Color(f'{color_variable}:N') if color_variable else alt.value('#1f77b4'),
        strokeDash=alt.StrokeDash(f'{linestyle_variable}:N') if linestyle_variable else alt.value([1, 0])
    )

    # Add error bars
    error_bars = alt.Chart(data).mark_errorbar(ticks=True).encode(
        x='n:Q',
        y='y:Q',
        yError=alt.YError('y_std:Q'),
        color=alt.Color(f'{color_variable}:N') if color_variable else alt.value('#1f77b4')
    )

    # Combine the main chart and error bars
    combined_chart = chart + error_bars

    # Add exponential fit lines
    for _, row in df.iterrows():
        with open(row.full_path.replace("stats.json", "bootstrap.json")) as f:
            p = yaml.load(f, Loader=loader)

        if not p:
            continue

        for p_ in p:
            x_exp = np.logspace(np.log10(128), np.log10(10**max_x), num=100)
            y_exp = p_[0] * np.power(x_exp, -p_[1]) + p_[2]
            exp_df = pd.DataFrame({'n': x_exp, 'y': y_exp})
            if color_variable:
                exp_df[color_variable] = getattr(row, color_variable)
            if linestyle_variable:
                exp_df[linestyle_variable] = getattr(row, linestyle_variable)

            exp_line = alt.Chart(exp_df).mark_line(opacity=0.2).encode(
                x='n:Q',
                y='y:Q',
                color=alt.Color(f'{color_variable}:N') if color_variable else alt.value('#1f77b4'),
                strokeDash=alt.StrokeDash(f'{linestyle_variable}:N') if linestyle_variable else alt.value([1, 0])
            )
            combined_chart += exp_line

    # Configure the chart
    combined_chart = combined_chart.properties(
        title=alt.Title(text=textwrap.fill(title, 90), anchor='middle'),
        width=600,
        height=400
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    ).configure_legend(
        orient='bottom',
        labelFontSize=10,
        titleFontSize=12
    )

    # Save the chart
    combined_chart.save(output_filename)


if __name__ == "__main__":
    plot(
        stats_file_list=snakemake.params.stats,
        output_filename=snakemake.output.plot,
        color_variable=snakemake.params.color_variable,
        linestyle_variable=snakemake.params.linestyle_variable,
        title=snakemake.params.title,
        max_x=snakemake.params.max_x,)
