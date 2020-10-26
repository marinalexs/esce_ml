import seaborn as sns
from matplotlib import pylab
import numpy as np
import pandas as pd

def hp_plot(df):
    df['log2_C'] = np.log2(df['C'])
    df['log2_gamma'] = np.log2(df['gamma'])

    fig, ax = pylab.subplots(1, 3, dpi=200, sharey='row')

    ax_ = ax[0]
    df_ = df[(df.model == 'linear')]
    sns.lineplot(x='log2_C', y='score_val', hue='n', data=df_, ax=ax_,
                     legend=False, ci='sd')
    ax_.set_ylabel(f'score_val')
    ax_.set_xlabel('linear, log2_C')

    ax_ = ax[1]
    df_ = df[(df.model == 'rbf')]
    df_ = df_.groupby(['log2_C', 's','n'])['score_val'].max().reset_index()
    sns.lineplot(x='log2_C', y='score_val', hue='n', data=df_, ax=ax_,
                     legend=False, ci='sd')
    ax_.set_xlabel('rbf, log2_C')

    ax_ = ax[2]
    df_ = df[(df.model == 'rbf')]
    df_ = df_.groupby(['log2_gamma', 's','n'])['score_val'].max().reset_index()
    sns.lineplot(x='log2_gamma', y='score_val', hue='n', data=df_, ax=ax_,
                     legend='full', ci='sd')
    ax_.set_xlabel('rbf, log2_gamma')

    pylab.plt.show()

def sc_plot(df):
    idx = df.groupby(['model', 'n'])['score_val'].idxmax()
    df_ = df.loc[idx]

    sns.lineplot(x='n', y='score_test', hue='model', data=df_, ci='sd')
    pylab.plt.xscale('log')
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