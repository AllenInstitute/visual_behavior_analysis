import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import os
import tensortools


def plot_TCA_factors(U_r, cells_df=[], cells_color_label=None, stim_df=[], trials_color_label="", cmap=None, psth_timebase=None):
    '''
    Plots TCA output. If specified, colors cells and trials by specified category.

    INPUTS:
    U_r                 output of tansorflow
    cells_df            output of processing.get_cells_df(oeids)
    cells_color_label   column in cells_df to use to color cells by
                        (currently, only options are "targeted_structure" and "imaging_depth"
    trials_df           output of processing.get_stimlulus_df(oeids)
    trials_color_label  column in trials_df to use to color trials by
    cmap                cmap for trial color
    psth_timebase       timebase for response factor profile

    RETURNS:
    fig             figure object
    '''
    num_factors = len(U_r.factors.factors)
    num_components = U_r.factors.rank
    factors_order = sort_by_temporal(U_r)

    fig, axes = plt.subplots(num_components, num_factors,
                             gridspec_kw={'width_ratios': [1.5, 1.5, 2]})
    fig.set_size_inches(15, 2.5 * num_components)

    # Set y axis the same for psth plots #
    psth_data = U_r.factors.factors[1]
    psth_min = np.min(psth_data)
    psth_max = np.max(psth_data)
    psth_range = [psth_min, psth_max]

    # create cells df with factors and cell description #
    df = pd.DataFrame(U_r.factors.factors[0])
    if cells_color_label is not None:
        cells = cells_df.join(df)
        if cells_color_label == "targeted_structure":
            cells_colormap = "nipy_spectral"
        else:
            cells_colormap = "Spectral"
    else:
        cells = df.copy()

    cells.sort_values(by=factors_order[0], inplace=True)
    cells = cells.reset_index(drop=True)
    cells = cells.reset_index()
    cells['cell_index'] = np.arange(0, cells.shape[0])

    # create trials df with factors and trials description #
    if trials_color_label is not None:
        df = pd.DataFrame(U_r.factors.factors[2])
        stim_df = stim_df.reset_index()
        trials = stim_df.join(df)
    else:
        trials = df.copy()
    trials['trial_index'] = np.arange(0, trials.shape[0])

    # iterate through factors and plot
    for i, ind_rank in enumerate(factors_order):

        # Plot cell factors, sorted
        cell_ax = axes[i, 0]
        sns.barplot(data=cells,
                    x='cell_index',
                    y=cells[ind_rank],
                    hue=cells_color_label,
                    palette=cells_colormap,
                    ax=cell_ax)
        change_width(cell_ax, .7)
        cell_ax.set_ylim(0, cells[0].max() * .5)  # cells[0].median())

        # Plot PSTH factors, sorted
        psth_ax = axes[i, 1]
        psth_fac = U_r.factors.factors[1][:, ind_rank]
        psth_ax.plot(psth_timebase, psth_fac, color='k')
        psth_ax.set_ylim(psth_range)

        # Plot trial factors
        trial_ax = axes[i, 2]
        sns.scatterplot(data=trials,
                        x="trial_index",
                        y=trials[ind_rank],
                        hue=trials_color_label,
                        palette=cmap,
                        ax=trial_ax)
        if i == 0:
            cell_ax.legend(fontsize=10)
            trial_ax.legend(fontsize=10)
        else:
            trial_ax.legend('')
            cell_ax.legend('')

        if i == len(factors_order) - 1:
            cell_ax.set_xlabel('cells')
            cell_ax.set_xticks(np.arange(1, cells.shape[0], np.round(cells.shape[0] * .20)))
            psth_ax.set_xlabel('time (s)')
            trial_ax.set_xlabel('trials')
        else:
            cell_ax.set_xlabel('')
            trial_ax.set_xlabel('')
            cell_ax.set_xticks([])
            psth_ax.set_xticks([])
            trial_ax.set_xticks([])

        plt.suptitle('{}'.format(trials_color_label), fontsize=20)

    return fig


def argsort_cells(U_r, ind_rank=0):
    '''
    Get a sorting order for the cell factors based off the cell weights for a particular rank.
    '''
    cell_weights = U_r.factors.factors[0][:, ind_rank]
    return np.argsort(cell_weights)


def sort_by_temporal(U_r):
    x = U_r.factors[1]
    maxbins = np.argmax(x, axis=0)
    sortorder = np.argsort(maxbins)
    return sortorder


def plot_responses(xr, title=''):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    for i in range(len(xr['trace_id'])):
        ax[0].plot(xr.coords['eventlocked_timestamps'].values,
                   xr.isel({'trace_id': i}).mean(dim='trial_id'))
        ax[0].set_xlabel('Time from stimulus onset (s)')
        ax[0].set_ylabel('filered events')
    ax[0].set_title('{} N = {}'.format(title, len(xr['trace_id'])))

    ax[1].plot(range(len(xr.coords['trace_id'].values)),
               xr.var(dim=['eventlocked_timestamps', 'trial_id']), 'o')
    ax[1].set_xlabel('Neurons')
    ax[1].set_ylabel('var(filtered events)')

    return fig


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def plot_fit_error(X, ranks=[1, 2, 3, 4, 5, 10, 15, 20]):
    print('this may take awhile....')
    fits = []
    errs = []
    for rank in ranks:
        U_r = tensortools.ncp_bcd(X, rank=rank, verbose=False)
        fits.append(U_r)
        errs.append(U_r.obj)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(ranks, errs, '-o')
    ax[0].set_ylabel('Final objective value')
    ax[0].set_xlabel('N factors')

    cmap = plt.cm.winter
    # optimization performance across iterations
    for i in range(0, len(ranks)):
        ax[1].plot(fits[i].obj_hist, '-', color=cmap(7 / (i + 1)))
        ax[1].set_xlabel('Optimization iteration')
        ax[1].set_ylabel('Objective (Fit error)')

    return fig


def plot_similarity_score(X, ranks=[1, 2, 3, 4, 5, 10, 20, 40, 60], n_runs=5):
    print('this may take awhile....')
    rank_similarity_scores = []
    for rank in ranks:
        U = []
        for n in range(n_runs):
            U_r = tensortools.ncp_bcd(X, rank=rank, verbose=False)
            U.append(U_r)

        similarity_scores = []
        for n in range(n_runs - 1):
            similarity = tensortools.kruskal_align(U[n].factors, U[n + 1].factors, permute_U=True, permute_V=True)
            similarity_scores.append(similarity)

        rank_similarity_scores.append(similarity_scores)

    rank_similarity_scores = np.array(rank_similarity_scores)

    # plot similarity scores
    fig = plt.figure(figsize=(7, 3))
    x = np.arange(len(ranks))
    sem = rank_similarity_scores.std(axis=1) / np.sqrt(rank_similarity_scores.shape[1])
    plt.errorbar(x, rank_similarity_scores.mean(axis=1), yerr=sem)
    plt.xticks(ticks=x, labels=ranks)
    plt.xlabel('N factors')
    plt.ylabel('similarity mean+/-SEM')

    return fig


def save_figure(fig, figsize=None, save_dir='', folder='', fig_name='', formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    if figsize is not None:
        fig.set_size_inches(figsize)
    fig.suptitle(fig_name)
    filename = os.path.join(fig_dir, fig_name)
    plt.tight_layout()
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape')
