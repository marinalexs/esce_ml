from esce.extrapolate import extrapolate

if __name__ == "__main__":
    extrapolate(
        snakemake.input.scores,
        snakemake.output.stats,
        snakemake.output.bootstraps,
        snakemake.params.bootstrap_repetitions,
    )
