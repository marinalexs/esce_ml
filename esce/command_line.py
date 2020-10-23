import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from esce.data import get_mnist, get_fashion_mnist, get_superconductivity
from esce.models import score_splits
from esce.sampling import split_grid
from esce.vis import hp_plot, sc_plot

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py

def load_dataset(data, label):
    data_path = Path(data)
    x = None
    y = None
    if data_path.suffix == ".h5":
        with h5py.File(data_path, "r") as f:
            x = f["/data"][...]
            y = f[f"/labels/{label}"][...]
    elif data_path.suffix == ".pkl":
        with data_path.open("rb") as f:
            d = pickle.load(f)
            x = d["data"]
            y = d[f"label_{label}"]
    else:
        raise ValueError("Unknown file format")
    return x,y    

def run(data, label, split, warm_start):
    x,y = load_dataset(data, label)
    with open(split, "rb") as f:
        splits = pickle.load(f)
    outfile = Path(f"results/{Path(split).stem}.csv")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    score_splits(outfile, x, y, splits, warm_start)

    # path = Path(f'results/{method}_{n_components}_{noise}.csv')
    # path.parent.mkdir(parents=True, exist_ok=True)
    # results.to_csv(path)

def datagen(dataset, method, n_components, noise=None, fmt="hdf5"):
    noise_str = "_n" + str(noise).replace(".", "_") if noise is not None else ""
    path = Path("data") / f"{dataset}_{method}{n_components}{noise_str}"
    if dataset == "mnist":
        x,y = get_mnist()
    elif dataset == "mnist_binary":
        x,y = get_mnist_binary()
    elif dataset == "fashion":
        x,y = get_fashion_mnist()
    elif dataset == "superconductivity":
        x,y = get_superconductivity()
    else:
        raise ValueError("Unknown dataset")

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
            if isinstance(y, dict):
                for k,v in y.items():
                    f.create_dataset(f"/labels/{k}", v.shape, data=v)
            else:
                f.create_dataset("/labels/default", y.shape, data=y)
    elif fmt == "pkl":
        path = path.with_suffix(".pkl")
        with path.open("wb") as f:
            d = {"data": x}
            if isinstance(y, dict):
                for k,v in y.items():
                    d[f"label_{k}"] = v
            else:
                d["label_default"] = y
            pickle.dump(d, f)
    else:
        raise ValueError("Unknown file format")
    print(f"Generated {dataset} data file '{path}'.")

def splitgen(data, label, seed, samples):
    path = Path(f"splits/{Path(data).stem}_{label}_s{seed}_{'_'.join(samples)}.split")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [int(sample) for sample in samples]

    _,y = load_dataset(data, label)
    splits = split_grid(y, n_samples=samples, seed=seed)

    with path.open("wb") as f:
        pickle.dump(splits, f)
    print(f"Generated split file '{path}'.")

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
    run_parser.add_argument('--label', default="default", type=str, help="which label to use")
    run_parser.add_argument('--split', type=str, help="split file to use", required=True)
    run_parser.add_argument('--warm', action="store_true", help="warm start")
    run_parser.set_defaults(run=True)

    datagen_parser.add_argument('dataset', default='mnist', type=str, help="dataset to use (mnist,fashion,superconductivity)")
    datagen_parser.add_argument('--method', default=None, type=str, help="dimensionality reduction method (pca,rp,tsne)")
    datagen_parser.add_argument('--components', default=2, type=int, help="number of components used in dimensionality reduction")
    datagen_parser.add_argument('--noise', default=None, type=float, help="whether or not to add noise")
    datagen_parser.add_argument('--format', default="hdf5", type=str, help="output file format (hdf5,pkl)")
    datagen_parser.set_defaults(datagen=True)

    splitgen_parser.add_argument("data", type=str, help="dataset file to use")
    splitgen_parser.add_argument("--label", default="default", type=str, help="label to use")
    splitgen_parser.add_argument("--seed", type=int, help="seed to use", required=True)
    splitgen_parser.add_argument("--samples", nargs="+", help="list number of samples", required=True)
    splitgen_parser.set_defaults(splitgen=True)

    viz_parser.add_argument('file', type=str, help="file containing the results to visualize")
    viz_parser.set_defaults(visualize=True)
    args = parser.parse_args()

    if args.run:
        run(args.data, args.label, args.split, args.warm)
    elif args.datagen:
        datagen(args.dataset, args.method, args.components, args.noise, args.format)
    elif args.splitgen:
        splitgen(args.data, args.label, args.seed, args.samples)
    elif args.visualize:
        visualize(args.file)

if __name__ == '__main__':
    main()