from esce.plot_hps import plot




if __name__ == "__main__":
    plot(
        stats_filename=snakemake.input.scores,
        output_filename=snakemake.output.plot,
        grid_filename=snakemake.params.grid,
        hyperparameter_scales=snakemake.params.hyperparameter_scales,
        model_name=snakemake.wildcards.model,
        title=snakemake.params.title,
    )
