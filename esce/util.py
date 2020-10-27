from pathlib import Path
import pickle
import h5py
from joblib import hash
import requests
from tqdm import tqdm

def download_file(url, path):
    """
    Download the URL and save it in the given path.

    Arguments:
        url: URL to download
        path: Path to store file to
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    bar = tqdm(total=total, unit='iB', unit_scale=True)
    block_size = 1024
    with path.open("wb") as f:
        for data in resp.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
    bar.close()
    if total != 0 and bar.n != total:
        raise ValueError

def pickled(root):
    """
    Decorator for caching using pickle.
    Stores a file in the specified directory.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    def decorator(fn):
        def wrapper(*args, **kwargs):
            h = hash(args) + hash(kwargs)
            path = root_path / f"{h}.pkl"
            if path.is_file():
                with path.open('rb') as f:
                    return pickle.load(f)
            else:
                out = fn(*args, **kwargs)
                with path.open('wb') as f:
                    pickle.dump(out, f)
                return out
        return wrapper
    return decorator

# TODO: add option to specify datatype in decorator?
def cached(root):
    """
    Decorator for HDF5 caching.
    Hashes arguments to provide a unique key.
    Stores the data in the specified path.
    Expects the returned data to be a floating point numpy array.
    The created dataset will be of type f32.
    """
    path = Path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = hash(args) + hash(kwargs)
            with h5py.File(path, 'a') as f:
                if key in f:
                    return f[key][...]
                else:
                    res = fn(*args, **kwargs)
                    f.create_dataset(key, res.shape, dtype='f', data=res)
                    return res
        return wrapper
    return decorator