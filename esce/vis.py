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
from pathlib import Path

pylab.rc('font', family='serif', serif='Times')
pylab.rc('xtick', labelsize=8)
pylab.rc('ytick', labelsize=8)
pylab.rc('axes', labelsize=8)

PLOT_PATH = Path("plots")

def hp_plot(df, grid, show=False):
    PLOT_PATH.mkdir(exist_ok=True)

    model_names = df["model"].unique()
    plots_per_model = {}
    for model_name in model_names:
        params = grid[model_name].keys()
        plots_per_model[model_name] = len(params)

    # Plots hyperparameters per model
    for model_name in model_names:
        df_ = df[df["model"] == model_name]
        if plots_per_model[model_name] == 0:
            continue

        names = df_.n
        palette = sns.color_palette("mako_r", n_colors=len(np.unique(names)))
        legend = [mpatches.Patch(color=i, label=j) for i, j in zip(palette, names)]

        fig, ax = pylab.subplots(1,plots_per_model[model_name], dpi=200, sharey='row', squeeze=False)
        for i, param_name in enumerate(grid[model_name].keys()):
            ax_ = ax[0,i]
            
            # Create small table, infer type back
            tmp = df_[["n", "acc_test"]].copy()
            tmp[param_name] = df_["params"].apply(lambda x: ast.literal_eval(x)[param_name])
            tmp.infer_objects()

            # Convert data to log if float
            if tmp[param_name].dtype in [np.float64, np.float32]:
                log_param = np.log2(tmp[param_name])
                param_name = "log " + param_name
                tmp[param_name] = log_param

            ax1 = sns.scatterplot(x=param_name, y='acc_test', hue='n', data=tmp, ax=ax_, ci='sd', palette=palette, legend=False)
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel(param_name)

        fig.subplots_adjust(bottom=0.2)

        pylab.plt.suptitle(MODEL_NAMES[model_name])
        pylab.figlegend(handles=legend, ncol=2, fontsize=8, loc='lower center', frameon=False)
        fig.savefig(PLOT_PATH / f'hp_{model_name}.png')
        if show:
            pylab.plt.show()
    
    pylab.plt.close("all")

def sc_plot(df, show=False):
    PLOT_PATH.mkdir(exist_ok=True)
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
    fig.savefig(PLOT_PATH / 'sc_plot.png')
    if show:
        pylab.plt.show()

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