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
from esce.grid import GRID
from esce.util import load_dataset, load_split, load_grid

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py 

def run(data_path, label, split_path, seeds, grid_name="default", warm_start=False):
    """
    Performs sample complexity computation.

    Arguments:
        data_path: Path to data file
        label: Label to use in data file
        split_path: Path to split file
        n_seeds: How many seeds to use
        grid: Grid to use
        warm_start: Whether or not to continue previous computation
    """

    x,y = load_dataset(data_path, label)
    found_seeds, splits = load_split(split_path)
    if found_seeds < len(seeds):
        raise ValueError(f"More speeds selected than available, found {found_seeds} seeds.")

    seeds = [int(seed) for seed in seeds]
    for seed in seeds:
        if seed >= found_seeds or seed < 0:
            raise ValueError(f"Invalid seed {seed}. Seed must be in [0,{found_seeds-1}].")

    if len(seeds) == 0:
        seeds = list(range(found_seeds))

    if grid_name in ["fine", "default", "coarse"]:
        grid = GRID[grid_name]
    else:
        grid = load_grid(grid_name)

    outfile = Path("results") / (split_path.stem + ".csv")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    score_splits(outfile, x, y, grid, splits, seeds, warm_start)

def datagen(dataset, method, n_components, noise=None, fmt="hdf5"):
    """
    Generates a data file.
    The file will be placed in the 'data' directory.

    Arguments:
        dataset: Pre-defined dataset to load
        method: Dimensionality reduction method to use
        n_components: Number of components used for dimensionality reduction
        noise: Noise factor
        fmt: Data file format (hdf5 or pkl)
    """

    noise_str = "_n" + str(noise).replace(".", "_") if noise is not None else ""
    path = Path("data") / f"{dataset}_{method}{n_components}{noise_str}"
    if dataset == "mnist":
        x,y = get_mnist()
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

def splitgen(data_path, label, n_seeds, samples):
    """
    Generates a split file.
    The file will be placed in the 'splits' directory.

    Arguments:
        data_path: Path to data file
        label: Label to use contained in data file
        n_seeds: Number of seeds to use in train_test_split
        samples: List of sample counts
    """
    path = Path("splits") / f"{data_path.stem}_{label}_s{n_seeds}_t{'_'.join(samples)}.split"
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [int(sample) for sample in samples]

    _,y = load_dataset(data_path, label)
    splits = split_grid(y, n_samples=samples, n_seeds=n_seeds)

    with path.open("wb") as f:
        pickle.dump((n_seeds, splits), f)
    print(f"Generated split file '{path}'.")

def visualize(path):
    if path.is_file():
        df = pd.read_csv(path, index_col=False)
        # hp_plot(df)
        sc_plot(df)
    else:
        frames = []
        for f in path.glob("*.csv"):
            frames.append(pd.read_csv(f, index_col=False))
        df = pd.concat(frames)
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
    run_parser.add_argument('--seeds', nargs="+", help="seeds to use", default=[])
    run_parser.add_argument('--grid', type=str, help="grid to use", default="default")
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
    splitgen_parser.add_argument("--seeds", type=int, help="number of seeds to use", required=True)
    splitgen_parser.add_argument("--samples", nargs="+", help="list number of samples", required=True)
    splitgen_parser.set_defaults(splitgen=True)

    viz_parser.add_argument('path', type=str, help="file/directory containing the results to visualize")
    viz_parser.set_defaults(visualize=True)
    args = parser.parse_args()

    if args.run:
        run(Path(args.data), args.label, Path(args.split), args.seeds, args.grid, args.warm)
    elif args.datagen:
        datagen(args.dataset, args.method, args.components, args.noise, args.format)
    elif args.splitgen:
        splitgen(Path(args.data), args.label, args.seeds, args.samples)
    elif args.visualize:
        visualize(Path(args.path))

if __name__ == '__main__':
    main()