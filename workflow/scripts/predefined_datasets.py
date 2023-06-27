from sklearn.datasets import fetch_openml


predefined_datasets = {
    'mnist': 
        {'features':{'pixel':lambda: fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)[0]},
         'targets':{'ten-digits':lambda: fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)[1].astype(int),
                    'odd-even':lambda: (fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)[1].astype(int) % 2).astype(int)},
        'covariates':{}
        }
}
