from pathlib import Path
import pickle
import h5py
from joblib import hash

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

def cached(root):
    """
    Decorator for HDF5 caching.
    Hashes arguments to provide a unique filename.
    Stores the data in the specified path.
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
                    dset = f.create_dataset(key, res.shape, dtype='f', data=res)
                    return dset[...]
        return wrapper
    return decorator