import seaborn as sns
from matplotlib import pylab
from matplotlib.ticker import ScalarFormatter
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
    model_names = df["model"].unique()
    plots_per_model = {}
    for model_name in model_names:
        params = GRID["default"][model_name].keys()
        plots_per_model[model_name] = len(params)
    total_plots = sum([v for k,v in plots_per_model.items()])

    root = np.sqrt(total_plots)
    if np.floor(root) != root:
        w = np.ceil(root)
        h = np.floor(root)
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
            y = i // w
            ax_ = ax[x][y]
            
            tmp = df_[["n", "acc_test"]].copy()
            tmp[param_name] = df_["params"].apply(lambda x: ast.literal_eval(x)[param_name])

            ax1 = sns.scatterplot(x=param_name, y='acc_test', hue='n', data=tmp, ax=ax_, legend=False, ci='sd')
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel(param_name)
            ax1.set_title(model_name)
            i += 1            

    pylab.plt.subplots_adjust(0.125, 0.1, 0.9, 0.9, 0.5, 1.0)
    pylab.plt.show()

def sc_plot(df):
    fig, ax = pylab.subplots(1, 1, dpi=200)

    models = list(df["model"].unique())

    names = [MODEL_NAMES[i] for i in models]
    palette = [MODEL_COLORS[i] for i in models]
    palette = sns.xkcd_palette(palette)
    legend = [mpatches.Patch(color=i, label=j) for i, j in zip(palette, names)]

    ax = sns.lineplot(x='n', y='acc_test', hue='model', data=df, ci='sd', palette=palette, ax=ax, legend=False)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Sample Size')
    ax.set_xticks(df.n.unique())
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0)

    fig.subplots_adjust(bottom=0.3)

    pylab.figlegend(handles=legend, ncol=2, fontsize=8, loc='lower center', frameon=False)
    pylab.plt.show()
    fig.savefig(f'sc_plot.png')

MODEL_COLORS = {
    "lda": "purple",
    "logit": "blue",
    "forest": "green",
    "ols": "pink",
    "lasso": "brown",
    "ridge": "red",
    "svm-linear": "light blue",
    "svm-rbf": "teal"
}