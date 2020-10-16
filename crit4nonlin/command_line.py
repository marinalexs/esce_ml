import argparse

from crit4nonlin.data import get_mnist
from crit4nonlin.models import score_splits
from crit4nonlin.sampling import split_grid

def run(method, n_components, noise=None, seeds=20):
    x, y = get_mnist(method=method, n_components=n_components, noise=noise)
    splits = split_grid(y, n_samples=(50, 100, 200, 500, 1000, 2000, 5000, 10000), n_seeds=seeds)
    results = score_splits(x, y, splits)
    results.to_csv(f'results/{method}_{n_components}_{noise}.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='mnist', type=str)
    parser.add_argument('--method', default='pca', type=str)
    parser.add_argument('--components', default=2, type=int)
    parser.add_argument('--seeds', default=20, type=int)
    args = parser.parse_args()

    print(args.method, args.components)

    run(args.method, args.components, None, args.seeds)

if __name__ == '__main__':
    main()