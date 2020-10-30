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

esce provides four different subcommands: *run,datagen,splitgen,visualize*.

The command *datagen* generates features and labels for a sample dataset and stores them into hdf5 files.
Additionally dimensionality reduction and noise can be added.

```
esce datagen mnist --method=pca --components=2
```

The command *splitgen* generates train, val and test splits for a given seed and a list of samples.
This is also stored as a file.

```
esce splitgen data/mnist_pca2.h5 --seeds=4 --samples 50 100 200 1000
```

Finally the sampling process can be started using the *run* command.

```
esce run data/mnist_pca2.h5 --label=default --split=splits/mnist_pca2_default_s10_t50_100_200.split
```

## Data file format

Data files are either stored in HDF5 or pickle files.
HDF5 files contain the datapoints in */data* and labels are stored in */labels/{label}*.
The default label for a dataset is stored in */labels/default*.

For pickle files, data and labels are stored in a dictionary.
The data is stored in "data" and the individual labels a stored in "label_{label}".

## Run unit tests

Tests can be run via the following command

```
python3 setup.py test
```
