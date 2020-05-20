import seaborn
from matplotlib import pylab
import numpy


# def hp_plot(df):
#     df['log2_C'] = numpy.log2(df['C'])
#     df['log2_gamma'] = numpy.log2(df['gamma'])
#
#     rows = df['n'].unique()
#     fig, ax = pylab.subplots(len(rows), 3, dpi=200, sharey='row')
#
#     for nj, j in enumerate(rows):
#         ax_ = ax[nj, 0]
#         df_ = df[(df.model == 'linear') & (df.n == j)]
#         seaborn.lineplot(x='log2_C', y='score_val', data=df_, ax=ax_,
#                          legend=False, ci='sd')
#         ax_.set_ylabel(f'n={j}')
#         if j == rows[-1]:
#             ax_.set_xlabel('linear, log2_C')
#         else:
#             ax_.set_xlabel('')
#
#         ax_ = ax[nj, 1]
#         df_ = df[(df.model == 'rbf') & (df.n == j)]
#         df_ = df_.groupby(['log2_C', 's'])['score_val'].max().reset_index()
#         seaborn.lineplot(x='log2_C', y='score_val', data=df_, ax=ax_,
#                          legend=False, ci='sd')
#         if j == rows[-1]:
#             ax_.set_xlabel('rbf, log2_C')
#         else:
#             ax_.set_xlabel('')
#
#         ax_ = ax[nj, 2]
#         df_ = df[(df.model == 'rbf') & (df.n == j)]
#         df_ = df_.groupby(['log2_gamma', 's'])['score_val'].max().reset_index()
#         seaborn.lineplot(x='log2_gamma', y='score_val', data=df_, ax=ax_,
#                          legend=False, ci='sd')
#         if j == rows[-1]:
#             ax_.set_xlabel('rbf, log2_gamma')
#         else:
#             ax_.set_xlabel('')
#
#     pylab.plt.show()


def hp_plot(df):
    df['log2_C'] = numpy.log2(df['C'])
    df['log2_gamma'] = numpy.log2(df['gamma'])

    fig, ax = pylab.subplots(1, 3, dpi=200, sharey='row')

    ax_ = ax[0]
    df_ = df[(df.model == 'linear')]
    seaborn.lineplot(x='log2_C', y='score_val', hue='n', data=df_, ax=ax_,
                     legend=False, ci='sd')
    ax_.set_ylabel(f'score_val')
    ax_.set_xlabel('linear, log2_C')

    ax_ = ax[1]
    df_ = df[(df.model == 'rbf')]
    df_ = df_.groupby(['log2_C', 's','n'])['score_val'].max().reset_index()
    seaborn.lineplot(x='log2_C', y='score_val', hue='n', data=df_, ax=ax_,
                     legend=False, ci='sd')
    ax_.set_xlabel('rbf, log2_C')

    ax_ = ax[2]
    df_ = df[(df.model == 'rbf')]
    df_ = df_.groupby(['log2_gamma', 's','n'])['score_val'].max().reset_index()
    seaborn.lineplot(x='log2_gamma', y='score_val', hue='n', data=df_, ax=ax_,
                     legend='full', ci='sd')
    ax_.set_xlabel('rbf, log2_gamma')

    pylab.plt.show()



def sc_plot(df):
    idx = df.groupby(['n', 's', 'model'])['score_val'].idxmax()
    df_ = df.loc[idx]
    seaborn.lineplot(x='n', y='score_test', hue='model', data=df_, ci='sd')
    pylab.plt.xscale('log')
    pylab.plt.show()


import pandas

df = pandas.read_csv('results/rand_4.csv')
hp_plot(df)
sc_plot(df)