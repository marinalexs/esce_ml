import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from esce.data import DATA
from esce.models import score_splits, MODELS, RegressionModel, precompute_kernels
from esce.sampling import split_grid
from esce.vis import hp_plot, sc_plot
from esce.grid import GRID, load_grid
from esce.util import load_dataset, load_split

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py
from sklearn.model_selection import ParameterGrid
from joblib import hash

def precomp(data_path, grid_name="default", include=None, exclude=None):
    x = load_dataset(data_path)
    grid = load_grid(grid_name)
    models = MODELS
    if include is not None:
        models = {k:v for k,v in models.items() if k in include}

    if exclude is not None:
        models = {k:v for k,v in models.items() if k not in exclude}

    precompute_kernels(x, models, grid)

def run(data_path, label, split_path, seeds, samples, grid_name="default", warm_start=False, include=None, exclude=None):
    """
    Performs sample complexity computation.

    Arguments:
        data_path: Path to data file
        label: Label to use in data file
        split_path: Path to split file
        seeds: Which seeds to use
        samples: Which samples to use
        grid: Grid to use / grid file to use
        warm_start: Whether or not to continue previous computation
    """

    x,y = load_dataset(data_path, label)
    found_seeds, splits = load_split(split_path)
    if found_seeds < len(seeds):
        raise ValueError(f"More speeds selected than available, found {found_seeds} seeds.")

    samples = [int(sample) for sample in samples]
    seeds = [int(seed) for seed in seeds]
    for seed in seeds:
        if seed >= found_seeds or seed < 0:
            raise ValueError(f"Invalid seed {seed}. Seed must be in [0,{found_seeds-1}].")

    if len(seeds) == 0:
        seeds = list(range(found_seeds))

    if len(samples) > 0:
        new_splits = dict()
        for sample in samples:
            new_splits[sample] = splits[sample]
        splits = new_splits

    models = MODELS
    if include is not None:
        models = {k:v for k,v in models.items() if k in include}

    if exclude is not None:
        models = {k:v for k,v in models.items() if k not in exclude}

    grid = load_grid(grid_name)
    outfile = Path("results") / (split_path.stem + ".csv")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    score_splits(outfile, x, y, models, grid, splits, seeds, warm_start)

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

    method_str = f"_{method}{n_components}" if method is not None else ""
    noise_str = "_n" + str(noise).replace(".", "_") if noise is not None else ""
    path = Path("data") / f"{dataset}{method_str}{noise_str}"
    if dataset not in DATA:
        raise ValueError("Unknown dataset")
    
    x, y = DATA[dataset]()
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

def retrieve(path, grid_name, output, show=None):
    grid = load_grid(grid_name)

    if path.is_dir():
        # Concat files into one, if not one
        frames = []
        for f in path.glob("*.csv"):
            frames.append(pd.read_csv(f, index_col=False))
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, index_col=False)

    # Select entries for a specified grid
    outer_frames = []
    for model_name in grid:
        rows_per_model = df[df["model"] == model_name]
        if rows_per_model.empty:
            continue
        
        model = MODELS[model_name]
        is_regression = isinstance(model, RegressionModel)

        # Select relevant grid
        inner_frames = []
        for params in ParameterGrid(grid[model_name]):
            param_hash = hash(params)
            df_ = rows_per_model[rows_per_model["param_hash"] == param_hash]
            inner_frames.append(df_)

        # Select best args
        df_ = pd.concat(inner_frames, ignore_index=True)

        if is_regression:
            idx = df_.groupby(['model', 'n'])['r2_val'].idxmax()
            df_ = df_.loc[idx]
        else:
            idx = df_.groupby(['model', 'n'])['acc_val'].idxmax()
            df_ = df_.loc[idx]

        df_.reset_index(drop=True, inplace=True)
        outer_frames.append(df_)

    sc_df = pd.concat(outer_frames, ignore_index=True)
    sc_df.reset_index(inplace=True, drop=True)
    print(sc_df)
    if output is not None:
        sc_df.to_csv(output)

    show_hp = show == "all" or show == "hp"
    show_sc = show == "all" or show == "sc"
    hp_plot(df, grid, show_hp)
    
    regr_missing = sc_df['acc_val'].isnull()
    sc_df.loc[regr_missing, 'acc_val'] = sc_df[regr_missing]['r2_val']
    sc_df.loc[regr_missing, 'acc_test'] = sc_df[regr_missing]['r2_test']
    sc_plot(sc_df, show_sc)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='functions')
    subparsers.required = True
    subparsers.dest = 'command'
    run_parser = subparsers.add_parser("run", help="perform estimation on dataset and split")
    datagen_parser = subparsers.add_parser("datagen", help="generate dataset file")
    splitgen_parser = subparsers.add_parser("splitgen", help="generate split file")
    retrieve_parser = subparsers.add_parser("retrieve", help="retrieve results")
    precomp_parser = subparsers.add_parser("precomp", help="precompute kernel matrices")
    parser.set_defaults(run=False, retrieve=False, datagen=False, splitgen=False, precomp=False)

    run_parser.add_argument('data', type=str, help="dataset file to use")
    run_parser.add_argument('--label', default="default", type=str, help="which label to use")
    run_parser.add_argument('--split', type=str, help="split file to use", required=True)
    run_parser.add_argument('--seeds', nargs="+", help="seeds to use", default=[])
    run_parser.add_argument('--samples', nargs="+", help="select a subset of samples", default=[])
    run_parser.add_argument('--grid', type=str, help="grid to use", default="default")
    run_parser.add_argument('--warm', action="store_true", help="warm start")
    run_parser.add_argument('--include', nargs="+", help="include only the specified models", default=None)
    run_parser.add_argument('--exclude', nargs="+", help="exclude models from comutation", default=None)
    run_parser.set_defaults(run=True)

    precomp_parser.add_argument('data', type=str, help="dataset file to use")
    precomp_parser.add_argument('--grid', type=str, help="grid to use", default="default")
    precomp_parser.add_argument('--include', nargs="+", help="include only the specified models", default=None)
    precomp_parser.add_argument('--exclude', nargs="+", help="exclude models from comutation", default=None)
    precomp_parser.set_defaults(precomp=True)

    datagen_parser.add_argument('dataset', default='mnist', type=str, help="dataset to use (mnist,fashion,superconductivity,higgs)")
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

    retrieve_parser.add_argument('path', type=str, help="file/directory containing the results to retrieve")
    retrieve_parser.add_argument('--grid', type=str, help="grid to analyse", default="default")
    retrieve_parser.add_argument('--show', type=str, help="show results (all/hp/sc)", default=None)
    retrieve_parser.add_argument('--output', type=str, help="output file location", default=None)
    retrieve_parser.set_defaults(retrieve=True)
    args = parser.parse_args()

    if args.run:
        run(Path(args.data), args.label, Path(args.split), args.seeds, args.samples, args.grid, args.warm, args.include, args.exclude)
    elif args.datagen:
        datagen(args.dataset, args.method, args.components, args.noise, args.format)
    elif args.splitgen:
        splitgen(Path(args.data), args.label, args.seeds, args.samples)
    elif args.retrieve:
        retrieve(Path(args.path), args.grid, args.output, args.show)
    elif args.precomp:
        precomp(Path(args.data), args.grid, args.include, args.exclude)

if __name__ == '__main__':
    main()