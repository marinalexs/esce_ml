import pandas as pd
import os
from pathlib import Path
import numpy as np
import scipy.optimize
from sklearn.linear_model import LinearRegression
import json


def confound_regression(data_path, confounds_path, out_path):
    data_raw = np.genfromtxt(data_path, delimiter=',')
    confounds = np.genfromtxt(confounds_path, delimiter=',')
    print(data_raw, confounds)
    print(data_raw.shape, confounds.shape)
    if len(data_raw.shape) == 1: data_raw=data_raw.reshape(-1,1)
    if len(confounds.shape) == 1: confounds=confounds.reshape(-1,1)
    assert len(data_raw) == len(confounds)

    model = LinearRegression()
    model.fit(confounds, data_raw)
    data_predicted = model.predict(confounds)
    data_corrected = data_raw - data_predicted

    np.savetxt(out_path,data_corrected, delimiter=',')

confound_regression(snakemake.input.features, snakemake.input.sampling, snakemake.output.features)
