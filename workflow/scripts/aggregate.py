from esce.aggregate import aggregate

if __name__ == "__main__":
    aggregate(snakemake.input.scores, snakemake.output.scores)
