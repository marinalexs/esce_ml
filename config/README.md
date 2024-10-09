# Configuration File

Note: Custom names should only use hyphens and never use underscores.


## General Configuration

This section sets parameters that will be used throughout the workflow for all evaluated machine learning scenarios. The most important parameters are:

- `sample_sizes`: Train set sizes you want to evaluate. Depending on your dataset size, you may need to change the default values.
- `seeds`: The number of times a model evaluation should be repeated on different subsets of the data. Start with a single seed to ensure everything works and then increase the number of seeds for higher quality results (10-20 is a reasonable default).

```yaml
val_test_frac: .25
# (float) Ratio of train set size to validation/test set size (in this case, 3:1:1)

val_test_min: False
# (bool or int) Minimum validation/test set size. Set to False to disable or an integer value to enforce a minimum size.

val_test_max: False
# (bool or int) Maximum validation/test set size. Set to False to disable or an integer value to enforce a maximum size.

bootstrap_repetitions: 100
# (int) Number of bootstrap repetitions for fitting power laws to learning curves.

stratify: False
# (bool) Stratify classes when splitting into train/val/test sets.

balanced: False
# (bool) In classification tasks, balance data via undersampling. This is a global setting that can be overridden in individual experiments.

quantile_transform: False
# (bool) Apply quantile transform to features and (regression-) targets. This is a global setting that can be overridden in individual experiments.

grid: "default"
# (str) Hyperparameter grid to use. This is a global setting that can be overridden in individual experiments.

seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# (list of int) Random seeds for splits. In this case, all analyses will be repeated 10 times with different (Monte Carlo) train/val/test splits.

sample_sizes: [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# (list of int) Train sample sizes to evaluate.
```

## Experiment Definitions

You can define multiple experiments, which are sets of feature sets, targets, covariates-of-no-interest, and machine learning models to be evaluated in every possible combination. Some combinations may not make sense, such as using the same models for regression and classification targets. In such cases, simply define multiple experiments.

```yaml
experiments:
# Defines experimental setups.
    mnist_benchmark:
    # (str) Name of first experiment.
        dataset: "mnist"
        # (str) Name of dataset to use.
        features: ["pixel",]
        # (list of str) List of feature sets to use.
        targets: ["ten-digits","odd-even"]
        # (list of str) List of target variables to use.
        confound_correction_method: ["correct-x", "correct-y", "correct-both", "matching", "with-cni", "only-cni", "none"]
        # (list of str) List of confound correction methods.
        confound_correction_cni: ["some-example-cni", "none"]
        # (list of str) List of covariate-of-no-interest datasets to use for confound correction.
        balanced: true
        # (bool) Override global balanced setting for this experiment.
        quantile_transform: true
        # (bool) Override global quantile_transform setting for this experiment.
        models: ["ridge-cls",]
        # (list of str) List of models to evaluate.
        grid: "custom"
        # (str) Override global grid setting for this experiment.
    another_set_of_experiments:
    # (str) Name of second experiment.
        [...]
```


## Models

### Classification

- majority-cls: Majority classifier
- logistic-regression-cls: Logistic regression classifier
- ridge-cls: Ridge classifier
- random-forest-cls: Random forest classifier
- xgb-cls: Gradient Boosting classifier
- rbf-kernel-svm-cls: Radial Basis Function kernel SVM classifier
- poly-kernel-svm-cls: Poly kernel SVM classifier
- pca-ridge-cls: Principal Component Analysis -> ridge classifier
- rfe-ridge-cls: Recursive Feature Elimination -> ridge classifier

### Regression

- mean-reg: Mean regressor
- ols-reg: Ordinary least squares regressor
- ridge-reg: Ridge regressor
- random-forest-reg: Random forest regressor
- xgb-reg: Gradient Boosting regressor
- rbf-kernel-svm-reg: Radial Basis Function kernel SVM regressor
- poly-kernel-svm-reg: Poly kernel SVM regressor
- pca-ridge-reg: Principal Component Analysis -> ridge regressor
- rfe-ridge-reg: Recursive Feature Elimination -> ridge regressor


## Dataset Definitions

In each dataset, you must provide three types of files:

1. Feature set files: The dimensions of the feature set files should be `n_samples x n_features`. The features will be standardized automatically.
2. Target files: The dimensions of the target files should be `n_samples x 1`.
3. Covariates-of-no-interest (optional): If you choose to include covariates-of-no-interest (CNI) files, their dimensions should be `n_samples x n_cni`. During matching, the CNIs will be standardized and weighted equally. If you require a different weighting, calculate the combined CNI values and save them as a single row CNI file.

Files should be saved in one of the following formats:
- Comma-separated values (`.csv`)
- Tab-separated values (`.tsv`)
- Numpy array file (`.npy`)

It's important to note that within a single dataset, all files must have the same number of rows and the same samples must be consistently referred to in all files.

```yaml
custom_datasets:
# Defines custom datasets
    dataset-name:
    # (str) Name of the first custom dataset
        features: {'NAME':'PATH_TO_CSV_TSV_NPY','NAME':'PATH_TO_CSV_TSV_NPY'}
        # (dict of str) Names and paths of the feature set files
        targets: {'NAME':'PATH_TO_CSV_TSV_NPY','NAME':'PATH_TO_CSV_TSV_NPY'}
        # (dict of str) Names and paths of the target files
        covariates: {'NAME':'PATH_TO_CSV_TSV_NPY','NAME':'PATH_TO_CSV_TSV_NPY'}
        # (dict of str) Names and paths of the covariate set files
    another-dataset-name:
    # (str) Name of the second custom dataset
        [...]
```

Note: All custom names (for datasets, features, targets, and covariates) should only use hyphens and never use underscores.