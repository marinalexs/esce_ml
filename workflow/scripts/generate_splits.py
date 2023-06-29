from esce.generate_splits import write_splitfile

if __name__ == "__main__":
    n_train = int(snakemake.wildcards.samplesize)
    n_val = n_test = min(
        round(n_train * snakemake.params.val_test_frac), snakemake.params.val_test_max
    ) if snakemake.params.val_test_max else round(n_train * snakemake.params.val_test_frac)
    n_val = n_test = max(n_val, snakemake.params.val_test_min) if snakemake.params.val_test_min else n_val
    assert n_train > 1 and n_val > 1 and n_test > 1


    write_splitfile(
        features_path=snakemake.input.features,
        targets_path=snakemake.input.targets,
        split_path=snakemake.output.split,
        sampling_path=snakemake.input.matching,
        sampling_type=snakemake.wildcards.matching,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=int(snakemake.wildcards.seed),
        stratify=snakemake.params.stratify,
    )
