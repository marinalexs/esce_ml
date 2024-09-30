import os
import textwrap
from pathlib import Path
import altair as alt

import pandas as pd
import yaml


def plot(
    stats_filename,
    output_filename,
    grid,
    hyperparameter_scales,
    model_name,
    title,
):
    """Plot the scores of a model.

    Args:
        stats_filename: path to the scores file
        output_filename: path to save the plot
        grid: grid of hyperparameters
        hyperparameter_scales: scales of the hyperparameters
        model_name: name of the model
        title: title of the plot
    """
    grid = grid[model_name]

    output_filename = Path(output_filename)
    # ignore empty files (insufficient samples in dataset)
    if os.stat(stats_filename).st_size == 0 or not grid:
        output_filename.touch()
        return

    scores = pd.read_csv(stats_filename)

    hp_names = list(grid.keys())
    metric = "r2_val" if "r2_val" in scores.columns else "acc_val"

    df = scores.melt(
        id_vars=["n", metric],
        value_vars=hp_names,
        var_name="hyperparameter",
        value_name="value"
    )

    df['value'] = df['value'].clip(lower=-1)

    # Create subplots for each hyperparameter
    charts = []
    for hp in hp_names:
        scale_type = hyperparameter_scales.get(hp, 'linear')

        base = alt.Chart(df[df['hyperparameter'] == hp]).mark_point().encode(
            x=alt.X('n:Q', scale=alt.Scale(type='log'), title='n'),
            y=alt.Y('value:Q', scale=alt.Scale(type=scale_type), title=hp),
            color=alt.Color(f'{metric}:Q', title=metric),
            tooltip=['n', 'value', metric]
        )

        # Add dotted lines for min and max values
        min_value = min(grid[hp])
        max_value = max(grid[hp])

        hline_min = alt.Chart(pd.DataFrame({'y': [min_value]})).mark_rule(
            strokeDash=[4, 4],
            color='black'
        ).encode(y='y:Q')

        hline_max = alt.Chart(pd.DataFrame({'y': [max_value]})).mark_rule(
            strokeDash=[4, 4],
            color='black'
        ).encode(y='y:Q')

        chart = (base + hline_min + hline_max).properties(title=hp)
        charts.append(chart)

    # Concatenate charts horizontally
    final_chart = alt.hconcat(*charts).resolve_scale(y='independent').properties(
        title=alt.Title(text=textwrap.fill(title, 90), anchor='middle')
    )

    # Save the chart
    final_chart.save(str(output_filename))




if __name__ == "__main__":
    plot(
        stats_filename=snakemake.input.scores,
        output_filename=snakemake.output.plot,
        grid=snakemake.params.grid,
        hyperparameter_scales=snakemake.params.hyperparameter_scales,
        model_name=snakemake.wildcards.model,
        title=snakemake.params.title,
    )
