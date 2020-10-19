import argparse

from esce.data import get_mnist
from esce.models import score_splits
from esce.sampling import split_grid
from esce.vis import hp_plot, sc_plot
# from matplotlib import pylab

def run(method, n_components, noise=None, seeds=20):
    x, y = get_mnist(method=method, n_components=n_components, noise=noise)
    splits = split_grid(y, n_samples=(50, 100, 200, 500, 1000, 2000, 5000, 10000), n_seeds=seeds)
    results = score_splits(x, y, splits)
    results.to_csv(f'results/{method}_{n_components}_{noise}.csv')

def visualize(path):
    df = pandas.read_csv(path)
    hp_plot(df)
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
    est_parser = subparsers.add_parser("estimate", help="perform estimation on dataset / model")
    viz_parser = subparsers.add_parser("visualize", help="visualize results")
    parser.set_defaults(estimate=False, visualize=False)

    est_parser.add_argument('--data', default='mnist', type=str, help="dataset to use")
    est_parser.add_argument('--method', default='pca', type=str, help="dimensionality reduction method")
    est_parser.add_argument('--components', default=2, type=int, help="number of components used in dimensionality reduction")
    est_parser.add_argument('--seeds', default=20, type=int, help="seed used for each split")
    est_parser.set_defaults(estimate=True)

    viz_parser.add_argument('--file', type=str, help="file containing the results to visualize")
    viz_parser.set_defaults(visualize=True)
    args = parser.parse_args()

    if args.estimate:
        print(args.method, args.components)
        run(args.method, args.components, None, args.seeds)

    elif args.visualize:
        visualize(args.file)

if __name__ == '__main__':
    main()