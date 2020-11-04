import seaborn as sns
from matplotlib import pylab
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from esce.models import MODEL_NAMES, MODELS
from esce.grid import GRID
import ast
from itertools import chain

pylab.rc('font', family='serif', serif='Times')
pylab.rc('xtick', labelsize=8)
pylab.rc('ytick', labelsize=8)
pylab.rc('axes', labelsize=8)

def hp_plot(df):
    # TODO fix grid
    df["params"] = [ast.literal_eval(x) for x in df["params"]]

    model_names = MODELS.keys()
    plots_per_model = {}
    for model_name in model_names:
        params = GRID["default"][model_name].keys()
        plots_per_model[model_name] = len(params)
    total_plots = sum([v for k,v in plots_per_model.items()])

    root = np.sqrt(total_plots)
    if np.floor(root) != root:
        w = np.ceil(root)
        h = np.foor(root)
    else:
        w = root
        h = root
    w, h = int(w), int(h)

    fig, ax = pylab.subplots(w,h, dpi=200, sharey='row')
    i = 0

    for model_name in model_names:
        df_ = df[df["model"] == model_name]
        if plots_per_model[model_name] == 0:
            continue

        for param_name in GRID["default"][model_name].keys():
            x = i % w
            y = i // h
            ax_ = ax[x][y]
            
            tmp = df_[["n", "acc_test"]].copy()
            tmp[param_name] = df_["params"].apply(lambda x: x[param_name])

            ax1 = sns.scatterplot(x=param_name, y='acc_test', hue='n', data=tmp, ax=ax_, legend=False, ci='sd')
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel(param_name)
            ax1.set_title(model_name)
            i += 1            

    pylab.plt.subplots_adjust(0.125, 0.1, 0.9, 0.9, 0.5, 1.5)
    pylab.plt.show()

def sc_plot(df):
    df["model"] = df["model"].apply(lambda x: MODEL_NAMES[x])

    ax = sns.lineplot(x='n', y='acc_test', hue='model', data=df, ci='sd')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Sample Size')
    pylab.plt.show()
