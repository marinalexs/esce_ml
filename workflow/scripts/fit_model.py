from esce.fit import fit
from esce.models import MODELS

if __name__ == "__main__":
    assert snakemake.wildcards.model in MODELS, "model not found"
    fit(
        snakemake.input.features,
        snakemake.input.targets,
        snakemake.input.split,
        snakemake.output.scores,
        snakemake.wildcards.model,
        snakemake.input.grid,
        snakemake.params.existing_scores,
    )
