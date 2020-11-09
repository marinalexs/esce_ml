# Empirical Sample Complexity Estimator (ESCE)

## Quickstart

### Install from GitHub

```
pip3 install git+https://github.com/maschulz/esce.git@master
```

### Install locally

Install the development package locally via

```
python3 setup.py develop --user
```

or install the package via

```
pip3 install .
```

## Usage:

esce provides five different subcommands: *run,datagen,splitgen,retrieve,precomp*.

The command *datagen* generates features and labels for a sample dataset and stores them into hdf5 files.
Additionally dimensionality reduction and noise can be added.

```
esce datagen mnist --method pca --components 8
```

The command *splitgen* generates train, val and test splits for a given seed and a list of samples.
This is also stored as a file.

```
esce splitgen data/mnist_pca8.h5 --seeds 4 --samples 50 100 200 1000
```

Finally the sampling process can be started using the *run* command.

```
esce run data/mnist_pca8.h5 --split splits/mnist_pca8_default_s10_t50_100_200.split
```

## Results & Visualization

To retrieve the accuracies on the test set or to visualize the results, use the *retrieve* command.

Example: Plot the results (scores and hyperparameters) in the given file for the grid default and write the scores to "out.csv".

```
esce retrieve results/mnist_pca8_default_s10_t100_200_500.csv --grid default --show all --output out.csv
```

Plots are automatically written to the `plots` directory.

## Kernel precomputation

The *precomp* subcommand allows to precompute the kernel gram matrices,
given a dataset, a hyperparameter grid and a list of models.

Example: Precompute the RBF kernel used by the SVM-RBF on the default grid for the MNIST dataset.

```
esce precomp data/mnist_pca8.h5 --grid default --include svm-rbf
```

## Data file format

Data files are either stored in HDF5 or pickle files.
HDF5 files contain the datapoints in */data* and labels are stored in */labels/{label}*.
The default label for a dataset is stored in */labels/default*.

For pickle files, data and labels are stored in a dictionary.
The data is stored in "data" and the individual labels a stored in "label_{label}".

Generated data files using *datagen* are stored in the `data` directory.

## Creating grid files

Grid files allow you to create custom model grids.
They are stored in YAML format and passed to the *run* subcommand via the `--grid` flag.

```yaml
ols: {}
lasso:
  alpha: [0.01, 0.1]
ridge:
  alpha: [0.02, 0.4]
svm-linear:
  C: [0.05, 0.3]
svm-rbf:
  C: [0.001, 0.2, 0.6]
  gamma: [0.2, 0.3, 0.6]
```

Multiple grids may be defined in the same file.
To select the grid, simply add `@<key>` to the file path,
for instance `grids/grid.yaml@grid1`

```yaml
grid1:
  ols: {}
  lasso:
    alpha: [0.01, 0.1]
  ridge:
    alpha: [0.02, 0.4]
  svm-linear:
    C: [0.05, 0.3]
  svm-rbf:
    C: [0.001, 0.2, 0.6]
    gamma: [0.2, 0.3, 0.6]

grid2:
  ols: {}
  lasso:
    alpha: [0.25]
  ridge:
    alpha: [0.02, 0.01]
  svm-linear:
    C: [0.02, 0.003]
  svm-rbf:
    C: [0.01, 0.2, 0.3]
    gamma: [0.2, 0.13, 0.6]
```

## Run unit tests

Tests can be run via the following command

```
python3 setup.py test
```
