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

# Configure YAML loader to correctly parse floating-point numbers
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
    list(u'-+0123456789.')
)
# This setup ensures that YAML correctly interprets floating-point numbers


def process_results(available_results: list) -> pd.DataFrame:
    """
    Read available results and return them as a pandas DataFrame.

    Args:
        available_results (list): List of file paths to the results files.

    Returns:
        pd.DataFrame: DataFrame containing processed results with extracted metadata.
    """
    df = pd.DataFrame(available_results, columns=["full_path"])
    # Extract metadata by parsing the file paths
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
    # Combine confound correction method and CNI into a single column
    df["cni"] = df[["confound_correction_method", "confound_correction_cni"]].agg("-".join, axis=1)
    return df


def plot(
    stats_file_list: list, 
    output_filename: str, 
    color_variable: Optional[str], 
    linestyle_variable: Optional[str], 
    title: str, 
    max_x: int = 6
):
    """
    Plot the results of a model using Altair.

    Args:
        stats_file_list (list): List of paths to the results files.
        output_filename (str): Path to save the generated plot.
        color_variable (Optional[str]): Variable to use for color encoding.
        linestyle_variable (Optional[str]): Variable to use for linestyle encoding.
        title (str): Title of the plot.
        max_x (int, optional): Maximum sample size in powers of 10 for exponential fit. Defaults to 6.
    """
    print(stats_file_list)
    # Process the results to extract metadata
    df = process_results(stats_file_list)

    data = []
    # Iterate over each result file to extract scoring metrics
    for _, row in df.iterrows():
        with open(row.full_path) as f:
            score = yaml.load(f, Loader=loader)
        if not score:
            print(row.full_path, "is empty - skipping")
            continue
                
        # Create a DataFrame for each score with mean and standard deviation
        df_ = pd.DataFrame(
            {"n": score["x"], "y": score["y_mean"], "y_std": score["y_std"]}
        )
        df_["model"] = row.model
        df_["features"] = row.features
        df_["target"] = row.target
        df_["cni"] = row.cni
        data.append(df_)

    # Combine all individual DataFrames into one
    if len(data) > 0:
        data = pd.concat(data, axis=0, ignore_index=True)
        data = data.sort_values(color_variable)
    else:
        # Create an empty file if there is no data to plot
        Path(output_filename).touch()
        return
    
    # Clip y-values to avoid extreme outliers in the plot
    data['y'] = data['y'].clip(lower=-1)
    data['y_std'] = data['y_std'].clip(upper=1)
    print(data)

    # Create the main line chart
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('n:Q', scale=alt.Scale(type='log')),
        y=alt.Y('y:Q', scale=alt.Scale(zero=True)),
        color=alt.Color(f'{color_variable}:N') if color_variable else alt.value('#1f77b4'),
        strokeDash=alt.StrokeDash(f'{linestyle_variable}:N') if linestyle_variable else alt.value([1, 0])
    )

    # Add error bars representing the standard deviation
    error_bars = alt.Chart(data).mark_errorbar(ticks=True).encode(
        x='n:Q',
        y='y:Q',
        yError=alt.YError('y_std:Q'),
        color=alt.Color(f'{color_variable}:N') if color_variable else alt.value('#1f77b4')
    )

    # Combine the main chart with error bars
    combined_chart = chart + error_bars

    # Add exponential fit lines from bootstrap data
    for _, row in df.iterrows():
        with open(row.full_path.replace("stats.json", "bootstrap.json")) as f:
            p = yaml.load(f, Loader=loader)

        if not p:
            continue

        for p_ in p:
            # Generate x-values for the exponential fit
            x_exp = np.logspace(np.log10(128), np.log10(10**max_x), num=100)
            # Compute y-values based on the fitted power law
            y_exp = p_[0] * np.power(x_exp, -p_[1]) + p_[2]
            exp_df = pd.DataFrame({'n': x_exp, 'y': y_exp})
            # Annotate with color and linestyle variables if provided
            if color_variable:
                exp_df[color_variable] = getattr(row, color_variable)
            if linestyle_variable:
                exp_df[linestyle_variable] = getattr(row, linestyle_variable)

            # Create the exponential fit line with reduced opacity
            exp_line = alt.Chart(exp_df).mark_line(opacity=0.2).encode(
                x='n:Q',
                y='y:Q',
                color=alt.Color(f'{color_variable}:N') if color_variable else alt.value('#1f77b4'),
                strokeDash=alt.StrokeDash(f'{linestyle_variable}:N') if linestyle_variable else alt.value([1, 0])
            )
            combined_chart += exp_line

    # Configure the final chart's appearance
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

    # Save the combined chart to the specified output file
    combined_chart.save(output_filename)


if __name__ == "__main__":
    """
    Entry point for the script when executed as a standalone program.
    Parses parameters from Snakemake and initiates the plotting process.
    """
    plot(
        stats_file_list=snakemake.params.stats,
        output_filename=snakemake.output.plot,
        color_variable=snakemake.params.color_variable,
        linestyle_variable=snakemake.params.linestyle_variable,
        title=snakemake.params.title,
        max_x=snakemake.params.max_x,
    )