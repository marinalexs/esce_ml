from esce.confound_regression import confound_regression

if __name__ == "__main__":
    confound_regression(
        snakemake.input.features, snakemake.input.confounds, snakemake.output.features
    )
