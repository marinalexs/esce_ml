import numpy as np
from sklearn.linear_model import LinearRegression


def confound_regression(data_path: str, confounds_path: str, out_path: str):
    data_raw = np.load(data_path)
    confounds = np.load(confounds_path)

    if len(data_raw.shape) == 1:
        data_raw = data_raw.reshape(-1, 1)
    if len(confounds.shape) == 1:
        confounds = confounds.reshape(-1, 1)
    assert len(data_raw) == len(confounds)

    x_mask = np.all(np.isfinite(data_raw), 1)
    y_mask = np.all(np.isfinite(confounds), 1)
    xy_mask = np.logical_and(x_mask, y_mask)

    model = LinearRegression()
    model.fit(confounds[xy_mask], data_raw[xy_mask])
    data_predicted = model.predict(confounds)
    data_corrected = data_raw - data_predicted

    np.save(out_path, data_corrected)


confound_regression(
    snakemake.input.features, snakemake.input.confounds, snakemake.output.features
)
