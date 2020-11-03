import seaborn as sns
from matplotlib import pylab
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from esce.models import MODEL_NAMES, MODELS
import ast
from itertools import chain

pylab.rc('font', family='serif', serif='Times')
pylab.rc('xtick', labelsize=8)
pylab.rc('ytick', labelsize=8)
pylab.rc('axes', labelsize=8)

def hp_plot(df):
    df["params"] = [ast.literal_eval(x) for x in df["params"]]
    hp_names = set().union(*df["params"])

    hp_table = pd.DataFrame(columns=list(hp_names))
    type_dict = {}
    for i, param in enumerate(df["params"]):
        for k,v in param.items():
            type_dict[k] = type(v)
            hp_table.loc[i, k] = v
    hp_table = hp_table.reindex(df.index)

    df = pd.concat([df, hp_table], axis=1)
    print(df)

    model_names = MODELS.keys()
    fig, ax = pylab.subplots(len(model_names), len(hp_names), dpi=200, sharey='row')
    i = 0

    for j, model_name in enumerate(model_names):
        df_ = df[df["model"] == model_name]

        for i, param_name in enumerate(hp_names):
            ax_ = ax[j][i]
            ax_.set_ylabel("Accuracy")
            ax_.set_xlabel(param_name)

            sns.scatterplot(x=param_name, y='acc_test', hue='n', data=df_, ax=ax_, legend=False, ci='sd')

    pylab.plt.show()

def sc_plot(df):
    ax = sns.lineplot(x='n', y='acc_test', hue='model', data=df, ci='sd')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Sample Size')
    pylab.plt.show()

# df = pandas.read_csv('results/pca_2_None.csv')
# hp_plot(df)
# sc_plot(df)

# from glob import glob
# F=glob('./results/pca_*_None.csv')
# df = pandas.DataFrame()

# for i,f in enumerate(F):

#     # ax = pylab.subplot(5,5,1+i)
#     # df = pandas.read_csv(f)
#     # sc_plot(df, ax)

# pylab.plt.show()