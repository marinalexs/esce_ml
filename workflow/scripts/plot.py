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
import traceback
import logging

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

# Set up logging
log_level = os.environ.get('ESCE_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_results(available_results: list) -> pd.DataFrame:
    """
    Read available results and return them as a pandas DataFrame.

    Args:
        available_results (list): List of file paths to the results files.

    Returns:
        pd.DataFrame: DataFrame containing processed results with extracted metadata.
    """
    logger.info(f"Processing {len(available_results)} result files")
    df = pd.DataFrame(available_results, columns=["full_path"])
    
    # Define expected columns
    expected_columns = [
        "dataset",
        "model",
        "features",
        "target",
        "confound_correction_method",
        "confound_correction_cni",
        "balanced",
        "grid",
    ]
    
    # Extract metadata by parsing the file paths
    def extract_metadata(path):
        parts = [Path(path).parts[1],Path(path).parts[3]]+Path(path).parts[4].split('_')
        logger.debug(f"Extracted parts: {Path(path).parts}")
        return pd.Series(parts + [None] * (len(expected_columns) - len(parts)), index=expected_columns)
    
    df[expected_columns] = df["full_path"].apply(extract_metadata)
    
    # Filter out empty files
    non_empty_files = []
    for file_path in df["full_path"]:
        try:
            with open(file_path) as f:
                content = yaml.load(f, Loader=yaml.SafeLoader)
            if content:  # Check if the file has content
                non_empty_files.append(file_path)
            else:
                logger.warning(f"{file_path} is empty - skipping")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
    
    # Keep only non-empty files
    df = df[df["full_path"].isin(non_empty_files)]
    
    # Combine confound correction method and CNI into a single column
    df["cni"] = df.apply(lambda row: f"{row['confound_correction_method']}-{row['confound_correction_cni']}" , axis=1)
    logger.info(f"Processed {len(df)} non-empty result files")
    logger.debug(f"DataFrame: {df.T}")
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
    logger.info(f"Starting plot generation for {len(stats_file_list)} files")
    logger.debug(f"Output filename: {output_filename}")
    logger.debug(f"Color variable: {color_variable}")
    logger.debug(f"Linestyle variable: {linestyle_variable}")

    # Process the results to extract metadata
    df = process_results(stats_file_list)

    data = []
    # Iterate over each result file to extract scoring metrics
    for _, row in df.iterrows():
        try:
            with open(row.full_path) as f:
                score = yaml.load(f, Loader=loader)
            if not score:
                logger.warning(f"{row.full_path} is empty - skipping")
                continue
                
            # Create a DataFrame for each score with mean and standard deviation
            df_ = pd.DataFrame(
                {"n": score["x"], "y": score["y_mean"], "y_std": score["y_std"]}
            )
            for col in ['model', 'features', 'target', 'cni', 'dataset']:
                if col in row.index:
                    df_[col] = row[col]
            data.append(df_)
        except Exception as e:
            logger.error(f"Error processing file {row.full_path}: {str(e)}")

    # Combine all individual DataFrames into one
    if len(data) > 0:
        data = pd.concat(data, axis=0, ignore_index=True)
        if color_variable and color_variable in data.columns:
            data = data.sort_values(color_variable)
    else:
        logger.warning("No data to plot. Creating an empty file.")
        Path(output_filename).touch()
        return

    # Convert 'y_std' column to numeric, replacing 'NaN' strings with actual NaN values
    data['y_std'] = pd.to_numeric(data['y_std'], errors='coerce')

    # Now clip the numeric values
    data['y_std'] = data['y_std'].clip(upper=1)

    # Replace NaN values in y_std with 0
    data['y_std'] = data['y_std'].fillna(0)

    logger.debug("Creating main line chart")
    # Create the main line chart
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('n:Q', scale=alt.Scale(type='log'), title='Sample Size'),
        y=alt.Y('y:Q', scale=alt.Scale(zero=False), title='Performance Metric'),
        color=alt.Color(f'{color_variable}:N', title=color_variable) if color_variable and color_variable in data.columns else alt.value('#1f77b4'),
        strokeDash=alt.StrokeDash(f'{linestyle_variable}:N', title=linestyle_variable) if linestyle_variable and linestyle_variable in data.columns else alt.value([1, 0])
    )

    logger.debug("Adding error bars")
    # Add error bars representing the standard deviation
    error_bars = alt.Chart(data).mark_errorbar(ticks=True).encode(
        x='n:Q',
        y='y:Q',
        yError=alt.YError('y_std:Q'),
        color=alt.Color(f'{color_variable}:N') if color_variable and color_variable in data.columns else alt.value('#1f77b4')
    )

    # Combine the main chart with error bars
    combined_chart = chart + error_bars

    logger.debug("Adding exponential fit lines")
    # Add exponential fit lines from bootstrap data
    for _, row in df.iterrows():
        bootstrap_file = row.full_path.replace("stats.json", "bootstrap.json")
        logger.debug(f"Processing bootstrap file: {bootstrap_file}")
        if not os.path.exists(bootstrap_file):
            logger.warning(f"Bootstrap file not found: {bootstrap_file}")
            continue
        
        with open(bootstrap_file) as f:
            p = yaml.load(f, Loader=loader)

        if not p:
            logger.warning(f"Empty bootstrap data in {bootstrap_file}")
            continue

        for p_ in p:
            logger.debug(f"Processing bootstrap parameters: {p_}")
            try:
                # Generate x-values for the exponential fit
                x_exp = np.logspace(np.log10(128), np.log10(10**max_x), num=100)
                # Compute y-values based on the fitted power law
                y_exp = float(p_[0]) * np.power(x_exp, -float(p_[1])) + float(p_[2])
            except Exception as e:
                logger.error(f"Error in exponential fit calculation: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue  # Skip this iteration if an error occurs
            
            exp_df = pd.DataFrame({'n': x_exp, 'y': y_exp})
            # Annotate with color and linestyle variables if provided
            if color_variable and color_variable in row.index:
                exp_df[color_variable] = row[color_variable]
            if linestyle_variable and linestyle_variable in row.index:
                exp_df[linestyle_variable] = row[linestyle_variable]

            # Create the exponential fit line with reduced opacity
            exp_line = alt.Chart(exp_df).mark_line(opacity=0.2).encode(
                x='n:Q',
                y='y:Q',
                color=alt.Color(f'{color_variable}:N') if color_variable and color_variable in exp_df.columns else alt.value('#1f77b4'),
                strokeDash=alt.StrokeDash(f'{linestyle_variable}:N') if linestyle_variable and linestyle_variable in exp_df.columns else alt.value([1, 0])
            )
            combined_chart += exp_line

    logger.debug("Configuring final chart appearance")
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
    )  # Removed .interactive()

    # Save the combined chart to the specified output file
    try:
        combined_chart.save(output_filename)
        logger.info(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        logger.error(f"Error saving plot to {output_filename}: {str(e)}")

if __name__ == "__main__":
    """
    Entry point for the script when executed as a standalone program.
    Parses parameters from Snakemake and initiates the plotting process.
    """
    logger.info("Starting plot generation script")
    plot(
        stats_file_list=snakemake.params.stats,
        output_filename=snakemake.output.plot,
        color_variable=snakemake.params.color_variable,
        linestyle_variable=snakemake.params.linestyle_variable,
        title=snakemake.params.title,
        max_x=snakemake.params.max_x,
    )
    logger.info("Plot generation script completed")