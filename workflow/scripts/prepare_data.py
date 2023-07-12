from esce.prepare_data import prepare_data

if __name__ == "__main__":
    prepare_data(
        snakemake.output.out,
        snakemake.wildcards.dataset,
        snakemake.wildcards.features_or_targets
        if hasattr(snakemake.wildcards, "features_or_targets")
        else "covariates",
        snakemake.wildcards.name,
        snakemake.params.custom_datasets,
    )
