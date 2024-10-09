import os
import textwrap
from pathlib import Path
import altair as alt
import pandas as pd
import yaml
import logging
import json

# Set up logging
log_level = os.environ.get('ESCE_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

def plot(
    stats_filename: str,
    output_filename: str,
    grid: dict,
    hyperparameter_scales: dict,
    model_name: str,
    title: str,
):
    """
    Plot the hyperparameter scores of a model.

    Args:
        stats_filename (str): Path to the CSV file containing scores.
        output_filename (str): Path to save the generated plot.
        grid (dict): Dictionary of hyperparameter grids used for the model.
        hyperparameter_scales (dict): Dictionary specifying the scale ('linear' or 'log') for each hyperparameter.
        model_name (str): Name of the model to plot.
        title (str): Title of the plot.
    """
    logging.info(f"Starting hyperparameter plot generation for {model_name}")
    
    # Extract the hyperparameter grid for the specified model
    grid = grid[model_name]

    output_filename = Path(output_filename)
    # Check if the scores file is empty or if no hyperparameters are provided
    if os.stat(stats_filename).st_size == 0 or not grid:
        logging.warning(f"Empty input file or no hyperparameters provided. Creating empty output file: {output_filename}")
        output_filename.touch()
        return

    # Load the scores into a DataFrame
    scores = pd.read_csv(stats_filename)

    # Check if the DataFrame is empty
    if scores.empty:
        logging.warning(f"Empty DataFrame. Creating empty output file: {output_filename}")
        output_filename.touch()
        return

    # List of hyperparameters to plot
    hp_names = list(grid.keys())
    # Determine the metric to use based on available columns
    metric = "r2_val" if "r2_val" in scores.columns else "bal_acc_val" if "bal_acc_val" in scores.columns else None

    if metric is None:
        logging.error("Neither 'r2_val' nor 'bal_acc_val' found in the DataFrame")
        raise ValueError("Neither 'r2_val' nor 'bal_acc_val' found in the DataFrame")

    logging.info(f"Using metric: {metric}")

    # Reshape the DataFrame for plotting
    df = scores.melt(
        id_vars=["n", metric],
        value_vars=hp_names,
        var_name="hyperparameter",
        value_name="value"
    )

    # Clip metric values to avoid extreme outliers in the plot
    df[metric] = df[metric].clip(lower=-1)

    # Initialize a list to hold individual hyperparameter charts
    charts = []
    for hp in hp_names:
        # Determine the scale type for the hyperparameter
        scale_type = hyperparameter_scales.get(hp, 'linear')
        logging.debug(f"Plotting hyperparameter {hp} with scale type: {scale_type}")

        # Create a scatter plot for the hyperparameter
        base = alt.Chart(df[df['hyperparameter'] == hp]).mark_point().encode(
            x=alt.X('n:Q', scale=alt.Scale(type='log'), title='n'),
            y=alt.Y('value:Q', scale=alt.Scale(type=scale_type), title=hp),
            color=alt.Color(f'{metric}:Q', title=metric),
            tooltip=['n', 'value', metric]
        )

        # Determine the minimum and maximum values for reference lines
        min_value = min(grid[hp])
        max_value = max(grid[hp])

        # Create dashed lines to indicate the hyperparameter's range
        hline_min = alt.Chart(pd.DataFrame({'y': [min_value]})).mark_rule(
            strokeDash=[4, 4],
            color='black'
        ).encode(y=alt.Y('y:Q', scale=alt.Scale(type=scale_type)))

        hline_max = alt.Chart(pd.DataFrame({'y': [max_value]})).mark_rule(
            strokeDash=[4, 4],
            color='black'
        ).encode(y=alt.Y('y:Q', scale=alt.Scale(type=scale_type)))

        # Combine the scatter plot with the reference lines
        chart = (base + hline_min + hline_max).properties(title=hp)
        charts.append(chart)

    # Concatenate all hyperparameter charts horizontally
    final_chart = alt.hconcat(*charts).resolve_scale(y='independent').properties(
        title=alt.Title(text=textwrap.fill(title, 90), anchor='middle')
    )

    # Save the final concatenated chart to the specified output file
    logging.info(f"Saving plot to {output_filename}")
    final_chart.save(str(output_filename))
    
    # For debugging purposes, save the chart specification as JSON
    with open(output_filename.with_suffix('.json'), 'w') as f:
        json.dump(final_chart.to_dict(), f, indent=2)
    
    logging.info("Plot generation completed successfully")

if __name__ == "__main__":
    """
    Entry point for the script when executed as a standalone program.
    Parses parameters from Snakemake and initiates the hyperparameter plotting process.
    """
    logging.info("Starting plot_hps script")
    plot(
        stats_filename=snakemake.input.scores,
        output_filename=snakemake.output.plot,
        grid=snakemake.params.grid,
        hyperparameter_scales=snakemake.params.hyperparameter_scales,
        model_name=snakemake.wildcards.model,
        title=snakemake.params.title,
    )
    logging.info("plot_hps script completed")