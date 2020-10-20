import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from esce.data import get_mnist
from esce.models import score_splits
from esce.sampling import split_grid
from esce.vis import hp_plot, sc_plot

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py

def run(data, label, seed):
    data_path = Path(data)
    x = None
    y = None
    if data_path.suffix == ".h5":
        with h5py.File(data_path, "r") as f:
            x = f["/data"][...]
            y = f[f"/labels/{label}"][...]
    elif data_path.suffix == ".pkl":
        with data_path.open("rb") as f:
            d = picke.load(f)
            x = d["data"]
            y = d[f"label_{label}"]

    # x, y = get_mnist(method=method, n_components=n_components, noise=noise)
    splits = split_grid(y, n_samples=(50, 100, 200, 500, 1000, 2000, 5000, 10000), seed=seed)
    results = score_splits(x, y, splits)

    path = Path(f'results/{method}_{n_components}_{noise}.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path)

def datagen(dataset, method, n_components, noise=None, fmt="hdf5"):
    path = Path("data")
    if dataset == "mnist":
        x, y = get_mnist()
        path = path / "mnist"
    else:
        raise ValueError

    if method == 'pca':
        x = PCA(n_components=n_components, whiten=True, random_state=0).fit_transform(x)
    elif method == 'rp':
        x = GaussianRandomProjection(n_components=n_components, random_state=0).fit_transform(x)
    elif method == "tsne":
        x = TSNE(n_components=n_components, random_state=0).fit_transform(x)
    x = StandardScaler().fit_transform(x)

    if noise is not None:
        x = (x + noise * np.random.randn(*x.shape)) / np.sqrt(1 + noise ** 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "hdf5":
        path = path.with_suffix(".h5")
        with h5py.File(path, 'w') as f:
            f.create_dataset("/data", x.shape, data=x)
            f.create_dataset("/labels/default", y.shape, data=y)
    elif fmt == "pkl":
        path = path.with_suffix(".pkl")
        with path.open("wb") as f:
            d = {"data": x, "label_default": y}
            pickle.dump(d, f)
    else:
        raise ValueError
    print(f"Generated {dataset} data file '{path}'.")

def splitgen():
    pass

def visualize(path):
    df = pd.read_csv(path)
    # hp_plot(df)
    sc_plot(df)

    # from glob import glob
    # F=glob('./results/pca_*_None.csv')
    # df = pandas.DataFrame()

    # for i,f in enumerate(F):
    #     ax = pylab.subplot(5,5,1+i)
    #     df = pandas.read_csv(f)
    #     sc_plot(df, ax)
    # pylab.plt.show()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='functions')
    subparsers.required = True
    subparsers.dest = 'command'
    run_parser = subparsers.add_parser("run", help="perform estimation on dataset and split")
    datagen_parser = subparsers.add_parser("datagen", help="generate dataset file")
    splitgen_parser = subparsers.add_parser("splitgen", help="generate split file")
    viz_parser = subparsers.add_parser("visualize", help="visualize results")
    parser.set_defaults(run=False, visualize=False, datagen=False, splitgen=False)

    run_parser.add_argument('data', type=str, help="dataset file to use")
    run_parser.add_argument('--seed', type=int, help="seed to use", required=True)
    run_parser.add_argument('--label', default="default", type=str, help="which label to use")
    run_parser.set_defaults(run=True)

    datagen_parser.add_argument('dataset', default='mnist', type=str, help="dataset to use")
    datagen_parser.add_argument('--method', default=None, type=str, help="dimensionality reduction method (pca,rp,tsne)")
    datagen_parser.add_argument('--components', default=2, type=int, help="number of components used in dimensionality reduction")
    datagen_parser.add_argument('--noise', default=None, type=float, help="whether or not to add noise")
    datagen_parser.add_argument('--format', default="hdf5", type=str, help="output file format (hdf5,pkl)")
    datagen_parser.set_defaults(datagen=True)

    # splitgen_parser.add_argument

    viz_parser.add_argument('file', type=str, help="file containing the results to visualize")
    viz_parser.set_defaults(visualize=True)
    args = parser.parse_args()

    if args.run:
        run(args.data, args.label, args.seed)

    elif args.datagen:
        datagen(args.dataset, args.method, args.components, args.noise, args.format)

    elif args.visualize:
        visualize(args.file)

if __name__ == '__main__':
    main()