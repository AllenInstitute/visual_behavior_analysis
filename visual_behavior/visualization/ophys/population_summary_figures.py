"""
Created on Thursday January 3 2019

@author: marinag
"""
import numpy as np
import matplotlib.pyplot as plt
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.visualization.utils import save_figure
import seaborn as sns
import pandas as pd

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white',
              {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': True, 'ytick.left': True, })
sns.set_palette('deep')


def plot_histogram(values, label, color='k', range=(0, 1), ax=None, offset=False, bins=30):
    results, edges = np.histogram(values, normed=True, range=range, bins=bins)
    binWidth = edges[1] - edges[0]
    if offset:
        ax.bar(edges[:-1] + binWidth, results * binWidth, binWidth, color=color, label=label, alpha=0.5)
    else:
        ax.bar(edges[:-1], results * binWidth, binWidth, color=color, label=label, alpha=0.5)
    return ax


# def plot_hist_for_metric(df, metric, condition='image_set', condition_values=['A','B','C','D'], colors=sns.color_palette(),
#                             cre_line=None, range=(0,1), ax=None, save_figures=False, save_dir=None, folder=None):
#     if ax is None:
#         figsize = (5,5)
#         fig, ax = plt.subplots(figsize=figsize)
#     ax1 = ax.twinx()
#     for i,condition_value in enumerate(condition_values):
#         values = df[df[condition]==condition_value][metric].values
# #         ax.hist(values, bins=30, label=condition_value, color=colors[i], range=range)
#         ax = plot_histogram(values, bins=30, label=condition_value, color=colors[i], range=range, ax=ax)
#         ax1 = sns.distplot(values, bins=30, kde=True, hist=False, color=colors[i], ax=ax1, kde_kws={'linewidth':3})
#     ax1.set_yticklabels('')
#     ax1.yaxis.set_ticks_position('none')
#     sns.despine(ax=ax1, right=True)
#     #     ax1.set_ylabel('density')
#     ax.set_xlabel(metric)
#     ax.set_ylabel('fraction of cells')
#     ax.set_title(cre_line)
#     ax.set_xlim(range[0]-0.1, range[1]+0.1)
# #     sns.despine(right=False)
#     # sns.despine(top=False)
#     if save_figures:
# #         fig.tight_layout()
# #         l = ax.legend(title=condition, fontsize='small')
# #         plt.setp(l.get_title(),fontsize='small')
#         plt.gcf().subplots_adjust(top=0.85)
#         plt.gcf().subplots_adjust(left=0.25)
#         plt.gcf().subplots_adjust(right=0.85)
#         plt.gcf().subplots_adjust(bottom=0.25)
#         save_figure(fig ,figsize, save_dir, folder, metric+'_by_'+condition+'_'+cre_line.split('-')[0]+'_hist')
#     return ax


def plot_hist_for_condition(df, metric, condition='image_set', condition_values=['A', 'B', 'C', 'D'],
                            colors=sns.color_palette(),
                            show_legend=False, show_kde=True, ax2_ylabel=None, label_kde=False,
                            cre_line=None, title=None, bins=30, range=(0, 1), ax=None, save_figures=False,
                            save_dir=None, folder=None):
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for i, condition_value in enumerate(condition_values):
        values = df[df[condition] == condition_value][metric].values
        ax = plot_histogram(values, bins=bins, label=condition_value, color=colors[i], range=range, ax=ax)
        if show_kde:
            ax1 = ax.twinx()
            ax1 = sns.distplot(values, bins=bins, kde=True, hist=False, color=colors[i], ax=ax1,
                               kde_kws={'linewidth': 3})
            if ax2_ylabel:
                ax1.set_ylabel(ax2_ylabel)
            else:
                ax1.set_ylabel('density')
            if not label_kde:
                ax1.set_yticklabels('')
                ax1.yaxis.set_ticks_position('none')
                sns.despine(ax=ax1, right=True)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    if title is None:
        ax.set_title('')
    else:
        ax.set_title(title)
    ax.set_xlim(range[0] - 0.1, range[1] + 0.1)
    sns.despine(right=False)
    if save_figures:
        if show_legend:
            legend = ax.legend(title=condition, fontsize='small')
            plt.setp(legend.get_title(), fontsize='small')
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(left=0.25)
        plt.gcf().subplots_adjust(right=0.85)
        plt.gcf().subplots_adjust(bottom=0.25)
        save_figure(fig, figsize, save_dir, folder,
                    metric + '_by_' + condition + '_' + cre_line.split('-')[0] + '_hist')
    return ax


def plot_hist_for_image_sets_cre_lines(df, metric, hist_ranges=[(-1, 1), (-1, 1)], xlabel=None, show_kde=True,
                                       label_kde=False, ax2_ylabel=None, show_legend=False,
                                       save_figures=False, save_dir=None, folder=None):
    condition = 'image_set'
    condition_values = ut.get_image_sets(df)
    colors = ut.get_colors_for_image_sets()
    cre_lines = ut.get_cre_lines(df)

    figsize = (4 * len(cre_lines), 4)
    fig, ax = plt.subplots(1, len(cre_lines), figsize=figsize, sharey=False)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_hist_for_condition(tmp, metric, condition, condition_values, colors, cre_line,
                                        range=hist_ranges[i],
                                        show_legend=False, show_kde=True, ax2_ylabel=None, label_kde=False,
                                        ax=ax[i], save_figures=False)
        if xlabel:
            ax[i].set_xlabel(xlabel)
        else:
            ax[i].set_xlabel(metric)
        ax[i].set_ylabel('')
        ax[i].set_title(cre_line)
    ax[0].set_ylabel('fraction of cells')
    plt.gcf().subplots_adjust(wspace=.38)
    plt.gcf().subplots_adjust(bottom=.25)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_hist')


def plot_cdf_for_condition(df, metric, condition='image_set', condition_values=['A', 'B', 'C', 'D'],
                           colors=sns.color_palette(),
                           cre_line=None, cdf_range=(0, 1), show_legend=True, ax=None, save_figures=False,
                           save_dir=None, folder=None):
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(cre_line)
    for i, condition_value in enumerate(condition_values):
        values = df[df[condition] == condition_value][metric].values
        ax = sns.distplot(values[~np.isnan(values)], hist=False, hist_kws={'cumulative': True},
                          kde_kws={'cumulative': True, 'linewidth': 2}, ax=ax, color=colors[i], label=condition_value)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    ax.set_xlim(cdf_range)

    if save_figures:
        if show_legend:
            legend = ax.legend(title=condition, fontsize='small')
            plt.setp(legend.get_title(), fontsize='small')
        #         fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(left=0.25)
        plt.gcf().subplots_adjust(right=0.85)
        plt.gcf().subplots_adjust(bottom=0.25)
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_' + cre_line.split('-')[0] + '_cdf')
    return ax


def plot_cdf_for_image_sets(df, metric, cdf_range=(0, 1), xlabel=None, show_legend=True,
                            save_figures=False, save_dir=None, folder=None):
    condition = 'image_set'
    condition_values = ut.get_image_sets(df)
    colors = ut.get_colors_for_image_sets()
    cre_lines = ut.get_cre_lines(df)

    figsize = (8, 3)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_cdf_for_condition(tmp, metric, condition, condition_values, colors=colors, cre_line=cre_line,
                                       cdf_range=cdf_range, ax=ax[i], save_figures=False, save_dir=save_dir,
                                       folder=folder)
        ax[i].legend_.remove()
        if xlabel is None:
            ax[i].set_xlabel(metric)
        else:
            ax[i].set_xlabel(xlabel)
    ax[1].set_ylabel('')
    if show_legend:
        legend = ax[i].legend(title=condition, fontsize='x-small')
        plt.setp(legend.get_title(), fontsize='x-small')
    if save_figures:
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(left=0.25)
        plt.gcf().subplots_adjust(right=0.85)
        plt.gcf().subplots_adjust(bottom=0.25)
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_' + '_cdf')


def adjust_box_widths(ax, fac):
    from matplotlib.patches import PathPatch
    # Adjust the withs of a seaborn-generated boxplot.
    for c in ax.get_children():
        if isinstance(c, PathPatch):
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5 * (xmin + xmax)
            xhalf = 0.5 * (xmax - xmin)
            xmin_new = xmid - fac * xhalf
            xmax_new = xmid + fac * xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
            for legend in ax.lines:
                if np.all(legend.get_xdata() == [xmin, xmax]):
                    legend.set_xdata([xmin_new, xmax_new])


def plot_boxplot_for_condition(df, metric, condition='image_set', condition_values=['A', 'B', 'C', 'D'],
                               colors=sns.color_palette(), hue='cre_line', ylabel=None,
                               range=(0, 1), ax=None, save_figures=False, save_dir=None, folder=None):
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=df, x=condition, y=metric,
                     hue=hue, ax=ax, width=0.4, dodge=True, palette=colors)
    if ylabel is None:
        ax.set_ylabel('fraction of cells per session')
    else:
        ax.set_ylabel(ylabel)
    ax.set_ylim(range[0] - 0.05, range[1] + .05)
    ax.get_legend().remove()
    ax.set_title(metric)
    sns.despine(offset=10, trim=True)

    if save_figures:
        #         fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(left=0.25)
        plt.gcf().subplots_adjust(right=0.85)
        plt.gcf().subplots_adjust(bottom=0.25)
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_box')
    return ax


def plot_boxplot_and_swarm_for_condition(df, metric, condition='cre_line', condition_values=ut.get_cre_lines,
                                         colors=sns.color_palette(), hue='image_set', ylabel=None, xlabel=None,
                                         title=None,
                                         plot_swarm=True, range=(0, 1), ax=None, save_figures=False, save_dir=None,
                                         folder=None, suffix=''):
    df[metric] = pd.to_numeric(df[metric])
    if ax is None:
        figsize = (4.5, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    hue_order = np.sort(df[hue].unique())
    ax = sns.boxplot(data=df, x=condition, y=metric, hue_order=hue_order,
                     hue=hue, ax=ax, dodge=True, palette=colors)  # color='white',
    adjust_box_widths(ax, 0.8)
    if plot_swarm:
        ax = sns.swarmplot(data=df, x=condition, y=metric,
                           size=3, ax=ax, hue=hue, hue_order=hue_order, color='.3', dodge=True)  # palette=colors,

        swarm_cols = ax.collections
        for swarm in swarm_cols:
            swarm.set_facecolors([0.6, 0.6, 0.6])
    #         swarm.set_linewidths([0.5])
    #         swarm.set_edgecolors([0.2,0.2,0.2])

    if ylabel is None:
        ax.set_ylabel('fraction of cells per session')
    else:
        ax.set_ylabel(ylabel)
    if xlabel is None:
        ax.set_xlabel(condition)
    else:
        ax.set_xlabel(xlabel)
    if title is None:
        ax.set_title(metric)
    else:
        ax.set_title(title)
    ax.set_ylim(range[0] - 0.05, range[1] + .05)
    ax.get_legend().remove()
    sns.despine(offset=10, trim=True)

    if save_figures:
        #         l = ax.legend(title=condition, fontsize='small')
        #         plt.setp(l.get_title(),fontsize='small')
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(left=0.25)
        plt.gcf().subplots_adjust(right=0.85)
        plt.gcf().subplots_adjust(bottom=0.25)
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_box_swarm' + suffix)

    return ax


def generate_figures_for_session_metric_image_sets(session_summary_df, metric, range=(0, 1), ylabel='fraction of cells',
                                                   title=None, plot_swarm=False, save_figures=False, save_dir=None,
                                                   folder=None):
    # # cre_line by image_sets
    # condition = 'image_set'
    # condition_values = ut.get_image_sets(session_summary_df)
    # hue = 'cre_line'
    # colors = ut.get_colors_for_cre_lines()
    #
    # df = session_summary_df.copy()
    # df = df[df.repeat == 1]
    # df[metric] = pd.to_numeric(df[metric])
    #
    # plot_boxplot_and_swarm_for_condition(df, metric, condition, condition_values, colors, hue, plot_swarm=True,
    #                                      range=range, ax=None, save_figures=save_figures, save_dir=save_dir,
    #                                      folder=folder)

    # image_sets by cre_line
    condition = 'cre_line'
    condition_values = ut.get_cre_lines(session_summary_df)
    hue = 'image_set'
    colors = ut.get_colors_for_image_sets()

    df = session_summary_df.copy()
    df = df[df.repeat == 1]
    df[metric] = pd.to_numeric(df[metric])
    df.cre_line = [cre_line.split('-')[0] for cre_line in df.cre_line.values]

    plot_boxplot_and_swarm_for_condition(df, metric, condition, condition_values, colors, hue, plot_swarm=plot_swarm,
                                         ylabel=None, xlabel=condition, title=title, range=range, ax=None,
                                         save_figures=save_figures,
                                         save_dir=save_dir, folder=folder)


def generate_figures_for_cell_summary_image_sets(cell_summary_df, metric, cdf_range=(-0.05, 1.05), xlabel=None,
                                                 hist_ranges=[(-5, 3), (-1, 1)], show_kde=True, show_legend=True,
                                                 save_figures=False, save_dir=None, folder=None):
    df = cell_summary_df[cell_summary_df.repeat == 1]
    plot_cdf_for_image_sets(df, metric, cdf_range=cdf_range, show_legend=show_legend, xlabel=xlabel,
                            save_figures=save_figures, save_dir=save_dir, folder=folder)

    # condition = 'image_set'
    # condition_values = ut.get_image_sets(df)
    # colors = ut.get_colors_for_image_sets()

    # for cre_line in cre_lines:
    #     df = cell_summary_df[cell_summary_df.cre_line==cre_line].copy()
    #     df = df[df.repeat==1]
    #
    #     plot_hist_for_condition(df, metric, condition, condition_values, colors, cre_line, show_legend=False, title=cre_line,
    #                         show_kde=True, range=range, ax=None, save_figures=save_figures, save_dir=save_dir, folder=folder)

    plot_hist_for_image_sets_cre_lines(df, metric, hist_ranges=hist_ranges, xlabel=None,
                                       show_legend=False, show_kde=True, ax2_ylabel=None, label_kde=False,
                                       save_figures=False, save_dir=None, folder=None)


def plot_mean_change_responses(df, vmax=0.3, colorbar=False, ax=None, save_dir=None, folder=None, use_events=False,
                               interval_sec=1, window=[-4, 8]):
    if use_events:
        vmax = 0.003
        label = 'mean event magnitude'
        suffix = '_events'
    else:
        # vmax = 0.3
        label = 'mean dF/F'
        suffix = ''
    image_set = df.image_set.unique()[0]
    cre_line = df.cre_line.unique()[0]
    if 'change_image_name' in df.keys():
        image_key = 'change_image_name'
        image_names = np.sort(df.change_image_name.unique())
        figsize = (20, 10)
    else:
        image_key = 'image_name'
        image_names = np.sort(df.image_name.unique())
        figsize = (12, 7)

    cells = []
    for image in image_names:
        tmp = df[(df[image_key] == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell_specimen_id.values[order])
        cells = cells + cell_ids
    if ax is None:
        fig, ax = plt.subplots(1, len(image_names), figsize=figsize, sharey=True, sharex=True)
        ax = ax.ravel()
    for i, image in enumerate(image_names):
        im_df = df[(df[image_key] == image)]
        len_trace = len(im_df.mean_trace.values[0])
        response_array = np.empty((len(cells), len_trace))
        for x, cell in enumerate(cells):
            tmp = im_df[im_df.cell_specimen_id == cell]
            if len(tmp) >= 1:
                trace = tmp.mean_trace.values[0]
            else:
                trace = np.empty((len_trace))
                trace[:] = np.nan
            response_array[x, :] = trace
        sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax[i], cmap='magma', cbar=colorbar,
                    cbar_kws={'label': label})
        xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=interval_sec, window=window)
        ax[i].set_xticks(xticks)
        if interval_sec < 1:
            ax[i].set_xticklabels(xticklabels)
        else:
            ax[i].set_xticklabels([int(x) for x in xticklabels])
        if response_array.shape[0] > 300:
            interval = 500
        else:
            interval = 50
        ax[i].set_xlim(0, (np.abs(window[0]) + window[1]) * 31.)
        ax[i].set_yticks(np.arange(0, response_array.shape[0], interval))
        ax[i].set_yticklabels(np.arange(0, response_array.shape[0], interval))
        ax[i].set_xlabel('time (sec)', fontsize=16)
        ax[i].set_title(image)
        ax[0].set_ylabel('cells')
    plt.suptitle('image set ' + image_set + '\n' + cre_line, x=0.52, y=.98, fontsize=18, horizontalalignment='center')
    fig.tight_layout()
    plt.gcf().subplots_adjust(top=0.85)
    if save_dir:
        save_figure(fig, figsize, save_dir, folder,
                    'change_response_matrix_' + cre_line + '_' + image_set + '_' + suffix)


def plot_tuning_curve_heatmap(df, vmax=0.3, sup_title=None, title=None, ax=None, save_dir=None, folder=None, use_events=False,
                              colorbar=True, include_omitted=False):
    if 'image_name' in df.keys():
        image_name = 'image_name'
        suffix = '_flashes'
        if ('omitted' in df.image_name.unique()) and (include_omitted is False):
            df = df[df.image_name != 'omitted']
    else:
        image_name = 'change_image_name'
        suffix = '_trials'
    if use_events:
        vmax = 0.03
        label = 'mean event magnitude'
        suffix = suffix + '_events'
    else:
        label = 'mean dF/F'
        suffix = suffix
    images = np.sort(df[image_name].unique())
    cell_list = []
    for image in images:
        tmp = df[(df[image_name] == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell_specimen_id.values[order])
        cell_list = cell_list + cell_ids
    response_matrix = np.empty((len(cell_list), len(images)))
    for i, cell in enumerate(cell_list):
        responses = []
        for image in images:
            response = df[(df.cell_specimen_id == cell) & (df[image_name] == image)].mean_response.values[0]
            responses.append(response)
        response_matrix[i, :] = np.asarray(responses)
    if ax is None:
        figsize = (5, 8)
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    ax = sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=vmax, robust=True, cbar=colorbar,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": label}, ax=ax)

    ax.set_title(title, va='bottom', ha='center')
    ax.set_xticklabels(images, rotation=90)
    ax.set_ylabel('cells')
    ax.set_yticks((0, response_matrix.shape[0]))
    ax.set_yticklabels((0, response_matrix.shape[0]), fontsize=14)
    if save_dir:
        plt.suptitle(sup_title, x=0.46, y=0.99, fontsize=18)
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.9)
        save_figure(fig, figsize, save_dir, folder, 'tuning_curve_heatmap_' + sup_title + '_' + title)
    return ax


def plot_pref_stim_responses(df, vmax=0.3, colorbar=False, ax=None, save_dir=None, folder=None, interval_sec=2,
                             use_events=False, window=[-4, 4], frame_rate=31.):
    if use_events:
        label = 'mean event magnitude'
        suffix = '_events'
    else:
        label = 'mean dF/F'
        suffix = ''
    image_set = df.image_set.unique()[0]
    cre_line = df.cre_line.unique()[0]
    if ax is None:
        figsize = (4, 7)
        fig, ax = plt.subplots(figsize=figsize)

    cdf = df[(df.pref_stim == True)]
    # cdf = df.copy()
    order = np.argsort(cdf.mean_response.values)[::-1]
    cells = list(cdf.cell_specimen_id.values[order])

    len_trace = len(cdf.mean_trace.values[0])
    response_array = np.empty((len(cells), len_trace))
    for x, cell in enumerate(cells):
        tmp = cdf[cdf.cell_specimen_id == cell]
        if len(tmp) >= 1:
            trace = tmp.mean_trace.values[0]
        else:
            trace = np.empty((len_trace))
            trace[:] = np.nan
        response_array[x, :] = trace
    sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax, cmap='magma', cbar=colorbar,
                cbar_kws={'label': label})
    xticks, xticklabels = sf.get_xticks_xticklabels(trace, frame_rate, interval_sec=interval_sec, window=window)
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(xticklabel) for xticklabel in xticklabels])
    # if response_array.shape[0] > 500:
    #     interval = 500
    # else:
    #     interval = 50
    ax.set_yticks((0, response_array.shape[0]))
    ax.set_yticklabels((0, response_array.shape[0]), fontsize=14)
    ax.set_xlabel('time after change (s)', fontsize=16)
    ax.set_title(cre_line)
    ax.set_ylabel('cells')
    if save_dir:
        plt.suptitle('image set ' + image_set, x=0.59, y=.99, fontsize=18)
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.9)
        save_figure(fig, figsize, save_dir, folder,
                    'pref_stim_response_matrix_' + cre_line + '_' + image_set + '_' + suffix)
    return ax


def plot_mean_response_by_repeat_heatmap(df, cre_line, title=None, ax=None, use_events=False, save_figures=False,
                                         save_dir=None, folder=None):
    # repeats = np.arange(1,11,1)
    repeats = np.sort(df.repeat.unique())
    df = df[(df.cre_line == cre_line) & (df.pref_stim == True) & (df.repeat.isin(repeats))]
    image_set = df.image_set.unique()[0]
    tmp = df[df.repeat == 1]
    #     tmp = cell_summary_df[cell_summary_df.cre_line==cre_line]

    if cre_line == 'Vip-IRES-Cre':
        interval = 100
    else:
        interval = 200
    if use_events:
        vmax = 0.03
        label = 'mean event magnitude'
        suffix = '_events'
    else:
        vmax = 0.3
        label = 'mean dF/F'
    order = np.argsort(tmp.mean_response.values)
    cell_ids = list(tmp.cell_specimen_id.values[order])
    cell_list = cell_ids
    # cell_list = cell_list + cell_ids
    #     cell_list = tmp.sort_values(by=['adaptation_index']).cell_specimen_id.values
    response_matrix = np.empty((len(cell_list), len(repeats)))
    for i, cell in enumerate(cell_list):
        responses = []
        for repeat in repeats:
            response = df[(df.cell_specimen_id == cell) & (df.repeat == repeat)].mean_response.values[0]
            responses.append(response)
        response_matrix[i, :] = np.asarray(responses)
    if ax is None:
        figsize = (5, 8)
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    ax = sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=vmax, robust=True, cbar=True,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": label}, ax=ax)
    ax.set_title(cre_line + '-' + image_set, va='bottom', ha='center')
    ax.set_xticklabels(repeats, rotation=90)
    ax.set_ylabel('cells')
    ax.set_xlabel('repeat')
    ax.set_yticks(np.arange(0, response_matrix.shape[0], interval))
    ax.set_yticklabels(np.arange(0, response_matrix.shape[0], interval))
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, 'repeat_response_heatmap_' + cre_line + '_' + image_set + suffix)
    return ax


def plot_flashes_on_trace(ax, trial_type=None, omitted=False, window=[-4, 4], alpha=0.15,
                          facecolor='gray', frame_rate=31.):
    stim_duration = .25
    blank_duration = .5
    change_frame = np.abs(window[0]) * frame_rate
    end_frame = (np.abs(window[0]) + window[1]) * frame_rate
    interval = blank_duration + stim_duration
    if omitted:
        array = np.arange((change_frame + interval), end_frame, interval * frame_rate)
        array = array[1:]
    else:
        array = np.arange(change_frame, end_frame, interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + (stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if trial_type == 'go':
        alpha = alpha * 3
    else:
        alpha
    array = np.arange(change_frame - ((blank_duration) * frame_rate), 0, -interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] - (stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_flashes_on_trace_time(ax, trial_type=None, omitted=False, flashes=False, window=[-3, 3], alpha=0.15,
                               facecolor='gray', frame_rate=31.):
    stim_duration = .25
    blank_duration = .5
    change_time = 0
    end_time = window[1]
    interval = blank_duration + stim_duration
    ax.set_xlim(window)
    if omitted:
        array = np.arange((change_time + interval), end_time + 0.1, interval)
    else:
        array = np.arange(change_time, end_time + 0.1, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + (stim_duration)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if trial_type == 'go':
        alpha = alpha * 3
    else:
        alpha
    array = np.arange(change_time - ((blank_duration)), window[0], -interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] - (stim_duration)
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_pre_stim_on_trace(ax, window=[-0.5, 0.75], alpha=0.3, facecolor='gray'):
    frame_rate = 31.
    stim_duration = .25
    # blank_duration = .5
    end_frame = int(np.abs(window[0]) * frame_rate)
    start_frame = int((np.abs(window[0]) - stim_duration) * frame_rate)
    ax.axvspan(start_frame, end_frame, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    # interval = int((blank_duration + stim_duration) * frame_rate)
    return ax


def plot_stim_on_trace(ax, window=[-0.5, 0.75], alpha=0.3, facecolor='gray'):
    frame_rate = 31.
    stim_duration = .25
    # blank_duration = .5
    start_frame = int(np.abs(window[0]) * frame_rate)
    end_frame = int((np.abs(window[0]) + stim_duration) * frame_rate)
    ax.axvspan(start_frame, end_frame, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    # interval = int((blank_duration + stim_duration) * frame_rate)
    return ax


def plot_mean_trace_from_mean_df(cell_data, frame_rate=31., ylabel='dF/F', legend_label=None, color='k', interval_sec=1,
                                 xlims=[-4, 4],
                                 ax=None, plot_sem=True, width=3):
    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    trace = cell_data.mean_trace.values[0]
    times = np.arange(0, len(trace), 1)
    sem = cell_data.sem_trace.values[0]
    ax.plot(trace, label=legend_label, linewidth=width, color=color)
    if plot_sem:
        ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)
    xticks, xticklabels = sf.get_xticks_xticklabels(trace, frame_rate, interval_sec, window=xlims)
    ax.set_xticks(xticks)
    if interval_sec >= 1:
        ax.set_xticklabels([int(x) for x in xticklabels])
    else:
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_mean_trace(mean_trace, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlims=[-4, 4],
                    ax=None):
    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    # times = np.arange(0, len(mean_trace), 1)
    ax.plot(mean_trace, label=legend_label, linewidth=3, color=color)
    xticks, xticklabels = sf.get_xticks_xticklabels(mean_trace, frame_rate, interval_sec, window=xlims)
    ax.set_xticks(xticks)
    if interval_sec < 1:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels([int(x) for x in xticklabels])
    ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
    ax.set_xlabel('time (sec)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_hist_for_cre_lines(df, metric, range=None, ax=None, save_figures=False, save_dir=None, folder=None,
                            offset=False):
    colors = ut.get_colors_for_cre_lines()
    if range is None:
        range = (0, 1)
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    cre_lines = ut.get_cre_lines(df)
    for i, cre_line in enumerate(cre_lines):
        values = df[(df.cre_line == cre_line)][metric].values
        if (offset == True) & (i > 0):
            ax = plot_histogram(values, label=cre_line, color=colors[i], range=range, ax=ax, offset=offset)
        else:
            ax = plot_histogram(values, label=cre_line, color=colors[i], range=range, ax=ax)
    ax.set_xlabel(metric)
    if 'correlation' in metric:
        ax.set_ylabel('fraction of pairs')
    else:
        ax.set_ylabel('fraction of cells')
    ax.legend()

    if save_figures:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, metric + '_across_cre_lines_hist')
    return ax


def plot_session_averages_for_cre_lines(metric, session_summary_df, ax=None, ylims=None, color_by_area=False,
                                        save_figures=False, save_dir=None, folder=None):
    if ylims:
        ylims = ylims
    else:
        ylims = (0, 1)
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    cre_lines = np.sort(session_summary_df.cre_line.unique())
    ax = sns.boxplot(data=session_summary_df, x='cre_line', y=metric, color='white', ax=ax, width=0.4, order=cre_lines)
    if color_by_area:
        #         area_colors = [ut.get_color_for_area(area) for area in np.sort(fdf.targeted_structure.unique())]
        area_colors = ut.get_colors_for_areas(session_summary_df)
        ax = sns.stripplot(data=session_summary_df, x='cre_line', y=metric, jitter=0.05, size=7, order=cre_lines,
                           hue='area', palette=area_colors, hue_order=cre_lines, ax=ax)
    else:
        cre_line_colors = ut.get_colors_for_cre_lines()
        ax = sns.stripplot(data=session_summary_df, x='cre_line', y=metric, jitter=0.05, size=7,
                           palette=cre_line_colors, hue='cre_line', hue_order=cre_lines, ax=ax, order=cre_lines)
    ax.get_legend().remove()
    ax.set_title('session averages')
    ax.set_ylim(ylims)

    if save_figures:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, metric + '_across_cre_lines_session_average')
    return ax


def plot_metric_data_for_cre_lines(metric, data, session_summary_df, range=None, ylims=None, color_by_area=False,
                                   save_figures=False, save_dir=None, folder=None, offset=False):
    figsize = (10, 5)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.ravel()
    ax[0] = plot_hist_for_cre_lines(data, metric, range=range, ax=ax[0], save_figures=False, offset=offset)
    ax[1] = plot_session_averages_for_cre_lines(metric, session_summary_df, ax=ax[1], ylims=ylims,
                                                color_by_area=color_by_area)
    fig.tight_layout()
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_across_cre_lines')
    return ax


def plot_hist_for_repeats(df, metric, range=None, cre_line=None, ax=None, save_figures=False, save_dir=None,
                          folder=None):
    colors = sns.color_palette()
    colors = ut.get_colors_for_image_sets()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    for i, repeat in enumerate(np.sort(df.repeat.unique())):
        values = df[df.repeat == repeat][metric].values
        ax = plot_histogram(values, label='repeat ' + str(int(repeat)),
                            color=colors[i], range=range, ax=ax)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    ax.legend()
    if save_figures:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, metric + '_by_flash_number_dist_' + cre_line.split('-')[0])
    return ax


def plot_cdf_for_repeats(df, metric, cre_line=None, ax=None, save_figures=False, save_dir=None, folder=None):
    colors = sns.color_palette()
    colors = ut.get_colors_for_image_sets()

    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    for i, repeat in enumerate(np.sort(df.repeat.unique())):
        values = df[df.repeat == repeat][metric].values
        ax = sns.distplot(values[~np.isnan(values)], hist=False, hist_kws={'cumulative': True}, color=colors[i],
                          kde_kws={'cumulative': True}, label=repeat, ax=ax)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    ax.legend()
    if save_figures:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, metric + '_by_flash_number_cdf_' + cre_line.split('-')[0])
    return ax


def plot_session_averages_for_repeats(session_summary_df, metric, ax=None, ylims=None, color_by_area=False,
                                      save_figures=False, save_dir=None, folder=None):
    if ylims:
        ylims = ylims
    else:
        ylims = (0, 1)
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    cre_line = session_summary_df.cre_line.unique()[0]
    ax = sns.boxplot(data=session_summary_df, x='repeat', y=metric, color='white', ax=ax, width=0.4)
    if color_by_area:
        area_colors = ut.get_colors_for_areas(session_summary_df)
        ax = sns.stripplot(data=session_summary_df, x='repeat', y=metric, jitter=0.05, size=7,
                           hue='area', palette=area_colors, ax=ax)
    else:
        colors = sns.color_palette()
        colors = ut.get_colors_for_image_sets()
        ax = sns.stripplot(data=session_summary_df, x='repeat', y=metric, jitter=0.05, size=7,
                           palette=colors, hue='repeat', ax=ax)
    ax.get_legend().remove()
    ax.set_title(cre_line)
    ax.set_ylim(ylims)

    if save_figures:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, metric + '_across_repeats_session_average_' + cre_line)
    return ax


def plot_violin_for_repeats(df, metric, cre_line=None, ax=None, save_figures=False, save_dir=None, folder=None):
    colors = sns.color_palette()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    ax = sns.violinplot(data=df, x='repeat', y=metric, color='white', ax=ax, cut=0 - 1)
    ax = sns.stripplot(data=df, x='repeat', y=metric, ax=ax, palette=colors)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_by_flash_number_violin_' + cre_line.split('-')[0])
    return ax


def plot_distributions_for_repeats(df, session_summary_df, metric, cre_line, save_dir=None, folder=None,
                                   save_figures=False, ymax=1.01, range=(0, 1)):
    image_set = df.image_set.unique()[0]
    figsize = (15, 5)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax = ax.ravel()
    ax[0] = plot_hist_for_repeats(df, metric, ax=ax[0], range=range)
    ax[0].set_title('cre_line')
    ax[1] = plot_cdf_for_repeats(df, metric, ax=ax[1])
    ax[1].set_title(cre_line)
    ax[2] = plot_session_averages_for_repeats(session_summary_df, metric, ax=ax[2], ylims=(-0.05, ymax),
                                              color_by_area=False)
    ax[2].set_title(cre_line)
    # ax[2] = plot_violin_for_repeats(df, metric, ax=ax[2])
    # plt.suptitle(cre_line, fontsize=16, x=0.52, y=1.0)
    fig.tight_layout()
    # plt.gcf().subplots_adjust(top=0.9)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder,
                    metric + '_by_flash_number_' + cre_line.split('-')[0] + '_' + image_set)


def plot_hist_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False, range=(0, 1)):
    colors = ut.get_colors_for_image_sets()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    for i, image_set in enumerate(np.sort(df.image_set.unique())):
        values = df[df.image_set == image_set][metric].values
        ax = plot_histogram(values, label=image_set, color=colors[i], range=range, ax=ax)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    ax.legend()
    #     if save_figures:
    #         fig.tight_layout()
    #         save_figure(fig ,figsize, save_dir, folder, metric+'_by_flash_number_dist_'+cre_line.split('-')[0])
    return ax


#
# def plot_cdf_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False):
#     colors = ut.get_colors_for_image_sets()
#     if ax is None:
#         figsize = (5,5)
#         fig, ax = plt.subplots(figsize=figsize)
#         ax.set_title(cre_line)
#     for i,image_set in enumerate(np.sort(df.image_set.unique())):
#         values = df[df.image_set==image_set][metric].values
#         ax = sns.distplot(values[~np.isnan(values)], hist=True, hist_kws={'cumulative':True,'histtype':'step'},
#                           kde_kws={'cumulative':True}, ax=ax, color=colors[i])
#     ax.set_xlabel(metric)
#     ax.set_ylabel('cumulative fraction')
#     ax.legend()
# #     if save_figures:
# #         fig.tight_layout()
# #         save_figure(fig ,figsize, save_dir, 'figure4', metric+'_by_flash_number_cdf_'+cre_line.split('-')[0])
#     return ax


def plot_violin_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False):
    colors = ut.get_colors_for_image_sets()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    ax = sns.violinplot(data=df, x='image_set', y=metric, color='white', ax=ax, cut=0 - 1)
    ax = sns.stripplot(data=df, x='image_set', y=metric, ax=ax, palette=colors)
    #     if save_figures:
    #         save_figure(fig ,figsize, save_dir, 'figure4', metric+'_by_flash_number_violin_'+cre_line.split('-')[0])
    return ax


def plot_session_average_for_image_sets(session_summary_df, metric, ax=None, ylims=None, color_by_area=False,
                                        save_figures=False, save_dir=None, folder=None):
    if ylims:
        ylims = ylims
    else:
        ylims = (0, 1)
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    cre_line = session_summary_df.cre_line.unique()[0]
    image_sets = ut.get_image_sets(session_summary_df)
    ax = sns.boxplot(data=session_summary_df, x='image_set', y=metric, color='white', ax=ax, width=0.4)
    if color_by_area:
        area_colors = ut.get_colors_for_areas(session_summary_df)
        ax = sns.stripplot(data=session_summary_df, x='image_set', y=metric, jitter=0.05, size=7,
                           hue='area', palette=area_colors, hue_order=image_sets, ax=ax)
    else:
        colors = ut.get_colors_for_image_sets()
        ax = sns.stripplot(data=session_summary_df, x='image_set', y=metric, jitter=0.05, size=7,
                           palette=colors, hue='image_set', ax=ax, hue_order=image_sets)
    ax.get_legend().remove()
    ax.set_title(cre_line)
    ax.set_ylim(ylims)

    if save_figures:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, metric + '_across_image_sets_session_average_' + cre_line)
    return ax


#
# def plot_session_average_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False, ymax=None, range=(0,1)):
#     if ax is None:
#         figsize = (6,5)
#         fig, ax = plt.subplots(figsize=figsize)
#         ax.set_title(cre_line)
#     image_sets = ut.get_image_sets(df)
#     for i,image_set in enumerate(np.sort(df.image_set.unique())):
#         idf = df[(df.image_set==image_set)]
#         for experiment_id in idf.experiment_id.unique():
#             edf = idf[(idf.experiment_id==experiment_id)]
#             area = edf.area.unique()[0]
#             color = ut.get_color_for_area(area)
#             mean = np.nanmean(edf[metric].values)
#             ax.plot(i ,mean, 'o', color=color)
#     ax.set_xlabel('image_set')
#     ax.set_ylabel(metric)
#     ax.set_title('session average')
#     ax.set_xticks(np.arange(0,4,1))
#     ax.set_xticklabels(image_sets)
#     if ymax is None:
#         y_min, y_max = ax.get_ylim()
#         ax.set_ylim([0,y_max+(0.1*y_max)])
#     else:
#         ax.set_ylim([0,ymax])
#     # make legend just for 2 points
#     for area in np.sort(df.area.unique())[::-1]:
#         experiment_id = idf[idf.area=='VISp'].experiment_id.unique()[0] #get one experiment for this area
#         mean = np.nanmean(df[df.experiment_id==experiment_id][metric].values)
#         color = ut.get_color_for_area(area)
#         ax.plot(0 ,mean, 'o', color=color, label=area)
#     ax.legend()
#     #     if save_figures:
#     #         fig.tight_layout()
#     #         save_figure(fig ,figsize, save_dir, 'figure4', metric+'_by_flash_number_cdf_'+cre_line.split('-')[0])
#     return ax


def plot_distributions_for_image_sets(df, session_summary_df, metric, cre_line, save_dir=None, folder=None,
                                      save_figures=False, ylims=(0, 1), range=(0, 1)):
    figsize = (15, 5)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax = ax.ravel()
    ax[0] = plot_hist_for_image_sets(df, metric, ax=ax[0], range=range)
    ax[1] = plot_cdf_for_image_sets(df, metric, ax=ax[1])
    # ax[2] = plot_violin_for_image_sets(df, metric, ax=ax[2])
    ax[2] = plot_session_average_for_image_sets(session_summary_df, metric, ax=ax[2], ylims=ylims)
    ax[2].set_title('session averages')
    plt.suptitle(cre_line, fontsize=16, x=0.52, y=1.0)
    fig.tight_layout()
    plt.gcf().subplots_adjust(top=0.9)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_by_image_set_' + cre_line.split('-')[0])


# def plot_mean_image_responses_flashes(data, cell_specimen_id, save_figures=False, save_dir=None, folder=None):
#     image_names = ut.get_image_names(data)
#     window = [-0.5, 0.75]
#     figsize = (15, 3)
#     fig, ax = plt.subplots(1, len(image_names), figsize=figsize, sharey=True)
#     for i, image_name in enumerate(image_names):
#         cell_data = data[(data.cell_specimen_id == cell_specimen_id) & (data.image_name == image_name)]
#         color = ut.get_color_for_image_name(image_names, image_name)
#         plot_cell_mean_trace_from_mean_df(cell_data, frame_rate=31., legend_label=None, color=color,
#                                           interval_sec=0.5, xlims=window, ax=ax[i])
#         ax[i] = plot_flashes_on_trace(ax[i], flashes=True, window=window, alpha=0.3)
#         ax[i].hlines(y=-0.05, xmin=np.abs(window[0]) * 31, xmax=(np.abs(window[0]) + 0.5) * 31.)
#         ax[i].set_title(image_name)
#         ax[i].set_ylabel('')
#     ax[0].set_ylabel('dF/F')
#     fig.tight_layout()
#
#     if save_figures:
#         save_figure(fig, figsize, save_dir, folder,
#                     str(int(cell_data.experiment_id)) + '_' + str(int(cell_specimen_id)))
#         plt.close()


def plot_change_repeat_response_pref_stim(cdf, cell_specimen_id, window=[-0.5, 0.75], save_figures=False,
                                          save_dir=None, folder=None, ax=None):
    cdf = cdf[cdf.pref_stim == True].copy()
    colors = ut.get_colors_for_changes()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for c, change in enumerate(np.sort(cdf.change.unique())):
        if change:
            label = 'change'
        else:
            label = 'repeat'
        tmp = cdf[cdf.change == change]
        trace = tmp.mean_trace.values[0]
        ax = plot_mean_trace_from_mean_df(tmp, 31., legend_label=label, color=colors[c],
                                          interval_sec=0.5, xlims=window, ax=ax)
    ax = plot_flashes_on_trace(ax, flashes=True, alpha=0.15, window=window)
    xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=0.5, window=window)
    ax.set_xticks(xticks)

    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0, (np.abs(window[0]) + window[1]) * 31.)
    ax.legend(bbox_to_anchor=(1.1, 1))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.2)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, str(int(cell_specimen_id)))
    return ax


def plot_change_repeat_response_all_stim(cdf, cell_specimen_id, window=[-0.5, 0.75], save_figures=False,
                                         save_dir=None, folder=None, ax=None):
    colors = ut.get_colors_for_changes()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for c, change in enumerate(np.sort(cdf.change.unique())):
        if change:
            label = 'change'
        else:
            label = 'repeat'
        tmp = cdf[cdf.change == change]
        traces = tmp.mean_trace.values
        ax = sf.plot_mean_trace(traces, 31., legend_label=label, color=colors[c], interval_sec=0.5, xlims=window, ax=ax)
    ax = plot_flashes_on_trace(ax, flashes=True, alpha=0.15, window=window)
    xticks, xticklabels = sf.get_xticks_xticklabels(np.mean(traces, axis=0), 31., interval_sec=0.5, window=window)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0, (np.abs(window[0]) + window[1]) * 31.)
    ax.legend(bbox_to_anchor=(1.1, 1))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.2)
    ax.set_ylim(0, ymax * 1.2)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder,
                    str(int(cdf.experiment_id.unique()[0])) + '_' + str(int(cell_specimen_id)))
        plt.close()
    return ax


def plot_change_repeat_response(mdf, cell_specimen_id, window=[-0.5, 0.75], save_figures=False,
                                save_dir=None, folder=None, ax=None):
    cdf = mdf[(mdf.cell_specimen_id == cell_specimen_id)].copy()
    figsize = (10, 5)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
    ax = ax.ravel()
    ax[0] = plot_change_repeat_response_pref_stim(cdf, cell_specimen_id, window=window, ax=ax[0])
    ax[0].set_title('preferred image')
    ax[0].legend_.remove()
    ax[1] = plot_change_repeat_response_all_stim(cdf, cell_specimen_id, window=window, ax=ax[1])
    ax[1].set_title('all images')
    ax[1].set_ylabel('')
    if save_figures:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder,
                    str(int(cdf.experiment_id.unique()[0])) + '_' + str(int(cell_specimen_id)))
        plt.close()
    return ax


def plot_population_response_across_conditions(df, condition='repeat', conditions=[1, 10],
                                               window=[-0.5, 0.75], save_figures=False, colors=None, autoscale=False,
                                               save_dir=None, folder=None, ax=None, pref_stim=True, omitted=False):
    image_set = df.image_set.unique()[0]
    # cre_line = df.cre_line.unique()[0]
    colors = colors[::-1]
    if pref_stim:
        df = df[df.pref_stim == True].copy()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    if np.abs(window[0]) >= 1:
        interval_sec = 1
    else:
        interval_sec = 0.5
    for c, condition_value in enumerate(conditions[::-1]):
        tmp = df[df[condition] == condition_value]
        # NOTE: The `get_image_color_lookup` function isn't defined. I'm commenting out this entire block to avoid linting errors. DRO, 5/28/20
        # if colors is None:
        #     image_lookup = get_image_color_lookup(mdf)
        #     image_names = df[df.image_set == image_set].image_name.unique()
        #     colors = get_colors_for_image_names(image_names, image_lookup)
        traces = tmp.mean_trace.values
        trace = np.mean(traces)
        ax = sf.plot_mean_trace(traces, 31., legend_label=condition_value, color=colors[c], interval_sec=interval_sec,
                                xlims=window, ax=ax)

    ax = plot_flashes_on_trace(ax, flashes=True, alpha=0.15, window=window, omitted=omitted)
    xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=interval_sec, window=window)
    ax.set_xticks(xticks)
    if interval_sec >= 1:
        ax.set_xticklabels([int(x) for x in xticklabels])
    else:
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(0, (np.abs(window[0]) + window[1]) * 31.)
    ax.legend(bbox_to_anchor=(1.1, 1), title=condition)
    if not autoscale:
        ymin, ymax = ax.get_ylim()
        if ymin > 0:
            ax.set_ylim(0, ymax * 1.2)
        else:
            ax.set_ylim(ymin * 1.2, ymax * 1.2)
    ax.set_title(image_set)
    if save_figures:
        assert False, 'save function not implemented'
        # fig.tight_layout()
        # save_figure(fig, figsize, save_dir, folder,
        #             str(int(cdf.experiment_id.unique()[0])) + '_' + str(int(cell_specimen_id)))
        # plt.close()
    return ax


def plot_population_response_all_flashes(idf, cells, title, filename, save_figures=False, save_dir=None, folder=None):
    cre_lines = ut.get_cre_lines(idf)
    colors = ut.get_colors_for_cre_lines()
    window = [-0.5, 0.75]

    figsize = (3.5, 5.5)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for i, cre_line in enumerate(cre_lines):
        traces = idf[
            (idf.cre_line == cre_line) & (idf.pref_stim == True) & (idf.cell_specimen_id.isin(cells))].mean_trace.values
        ax[i] = sf.plot_mean_trace(traces, 31., color=colors[i], interval_sec=0.5,
                                   xlims=window, ax=ax[i])
        ax[i] = plot_flashes_on_trace(ax[i], flashes=True, window=window)
        ax[i].set_title(cre_line)
        ax[i].set_ylim(ymin=0)
        fig.tight_layout()
    ax[0].set_xlabel('')
    plt.suptitle(title, x=.6, y=0.9, fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(top=0.8)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, filename)
    # return ax


def plot_omitted_flash_response_all_stim(odf, cell_specimen_id, ax=None, save_dir=None, window=[-2, 3], legend=False):
    cdf = odf[odf.cell_specimen_id == cell_specimen_id]
    image_names = np.sort(cdf.image_name.unique())
    if ax is None:
        figsize = (7, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for image_name in image_names:
        color = ut.get_color_for_image_name(image_names, image_name)
        ax = plot_mean_trace_from_mean_df(cdf[cdf.image_name == image_name],
                                          31., legend_label=None, color=color,
                                          interval_sec=1, xlims=window, ax=ax)
    ax = plot_flashes_on_trace(ax, trial_type=None, alpha=0.3, window=window, omitted=True, flashes=False)
    ax.set_xlabel('time (sec)')
    ax.set_title('omitted flash response')
    if legend:
        ax.legend(loc=9, bbox_to_anchor=(1.3, 1.3))
    if save_dir:
        fig.tight_layout()
        save_figure(fig, (6, 5), save_dir, 'omitted_flash_response', str(cell_specimen_id))
        plt.close()
    return ax


def plot_image_change_response(tdf, cell_specimen_id, legend=True, save=False, ax=None, window=[-2, 3]):
    ylabel = 'mean dF/F'
    cdf = tdf[tdf.cell_specimen_id == cell_specimen_id]
    images = np.sort(cdf.change_image_name.unique())
    images = images[images != 'omitted']
    if ax is None:
        figsize = (7, 5)
        fig, ax = plt.subplots(figsize=figsize)
    for c, change_image_name in enumerate(images):
        color = ut.get_color_for_image_name(images, change_image_name)
        ax = plot_mean_trace_from_mean_df(cdf[cdf.change_image_name == change_image_name],
                                          31., legend_label=None, color=color,
                                          interval_sec=1, xlims=window, ax=ax)
    ax = plot_flashes_on_trace(ax, trial_type='go', alpha=0.3, window=window)
    ax.set_title('cell_specimen_id: ' + str(cell_specimen_id))
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(images, loc=9, bbox_to_anchor=(1.19, 1))
    return ax


def plot_change_omitted_responses(tdf, odf, cell_specimen_id, save_figures=False, save_dir=None, folder=None):
    figsize = (10, 4)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
    ax = ax.ravel()

    cdf = tdf[tdf.cell_specimen_id == cell_specimen_id]
    cre_line = cdf.cre_line.unique()[0]
    image_set = cdf.image_set.unique()[0]
    area = cdf.image_set.unique()[0]
    ax[0] = plot_image_change_response(tdf, cell_specimen_id, legend=False, save=False, ax=ax[0], window=[-4, 8])
    ax[0].set_xlim(1 * 31, 7 * 31)
    ax[0].set_title('change flash response')
    ax[1] = plot_omitted_flash_response_all_stim(odf, cell_specimen_id, ax=ax[1], window=[-3, 3])
    ax[1].set_ylabel('')
    fig.tight_layout()

    if save_figures:
        fig.tight_layout()
        filename = str(int(cdf.experiment_id.unique()[0])) + '_' + str(
            int(cell_specimen_id)) + '_' + cre_line + '_' + area + '_' + image_set
        save_figure(fig, figsize, save_dir, folder, filename)
        plt.close()


def plot_average_flash_response_example_cells(analysis, active_cell_indices, save_figures=False, save_dir=None,
                                              folder=None, ax=None):
    dataset = analysis.dataset
    fdf = analysis.stimulus_response_df.copy()
    last_flash = fdf.flash_number.unique()[-1]  # sometimes last flash is truncated
    fdf = fdf[fdf.flash_number != last_flash]

    conditions = ['cell_specimen_id', 'image_name']
    mdf = ut.get_mean_df(fdf, analysis, conditions=conditions, flashes=True)

    cell_specimen_ids = [dataset.get_cell_specimen_id_for_cell_index(cell_index) for cell_index in active_cell_indices]

    image_names = np.sort(mdf.image_name.unique())

    if ax is None:
        if len(active_cell_indices) < 10:
            figsize = (12, 8.3)
        else:
            figsize = (12, 10)
        fig, ax = plt.subplots(len(cell_specimen_ids), len(image_names), figsize=figsize, sharex=True)
        ax = ax.ravel()

    i = 0
    for c, cell_specimen_id in enumerate(cell_specimen_ids):
        cell_data = mdf[(mdf.cell_specimen_id == cell_specimen_id)]
        maxs = [np.amax(trace) for trace in cell_data.mean_trace.values]
        ymax = np.amax(maxs) * 1.2
        for m, image_name in enumerate(image_names):
            cdf = cell_data[(cell_data.image_name == image_name)]
            color = ut.get_color_for_image_name(image_names, image_name)
            ax[i] = plot_mean_trace_from_mean_df(cdf, 31., color=[.5, .5, .5], interval_sec=0.5,
                                                 xlims=analysis.flash_window, ax=ax[i])
            ax[i] = plot_flashes_on_trace(ax[i], window=analysis.flash_window, facecolor=color, alpha=0.3)
            if 'Vip' in dataset.metadata.cre_line.values[0]:
                ax[i].vlines(x=0, ymin=0, ymax=.25, linewidth=3)
            elif 'Slc' in dataset.metadata.cre_line.values[0]:
                ax[i].vlines(x=0, ymin=0, ymax=.1, linewidth=3)
            #         sns.despine(ax=ax[i])
            ax[i].axis('off')
            ax[i].set_ylim(-0.05, ymax)
            if m == 0:
                ax[i].set_ylabel('x')
            if c == 0:
                ax[i].set_title(image_name)
            if c == len(cell_specimen_ids):
                ax[i].set_xlabel('time (s)')
            i += 1

    # fig.tight_layout()
    if save_figures:
        if save_dir:
            save_figure(fig, figsize, save_dir, folder, dataset.analysis_folder)
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'example_traces_all_flashes', dataset.analysis_folder)
        plt.close()


def plot_fraction_cells_over_threshold(cell_df, metric, threshold, condition='image_set',
                                       condition_values=['A', 'B', 'C', 'D'],
                                       colors=sns.color_palette(), less_than=False, title=None, sharey=True, ymax=None,
                                       save_figures=False, save_dir=None, folder=None, suffix=''):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (6, 3)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)]
            if c == 0:
                x = 0
            else:
                x = x + 0.2
            if less_than:
                tmp = t[(t[metric] < threshold)]
            else:
                tmp = t[(t[metric] > threshold)]
            fraction = len(tmp) / float(len(t))
            ax[i].bar(x, fraction, width=0.1, color=colors[c])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0, len(condition_values) * 0.2, 0.2))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_xlabel('image set')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[0].set_ylabel('fraction of cells')
    if title:
        plt.suptitle(title, x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    elif less_than:
        plt.suptitle('fraction of cells with ' + metric + ' < ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    else:
        plt.suptitle('fraction of cells with ' + metric + ' > ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15)
    #     plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.25)
    if sharey:
        plt.gcf().subplots_adjust(wspace=.2)
    else:
        plt.gcf().subplots_adjust(wspace=.3)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_' + condition + '_fraction' + suffix)


def plot_fraction_cells_over_threshold_stacked(cell_df, metric, threshold, condition='image_set',
                                               condition_values=['A', 'B', 'C', 'D'], colors=sns.color_palette(),
                                               less_than=False, ymax=None,
                                               suffix='', sharey=True, title=None, save_figures=False, save_dir=None,
                                               folder=None):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (3.5, 6)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)]
            if c == 0:
                x = 0
            else:
                x = x + 0.2
            if less_than:
                tmp = t[(t[metric] < threshold)]
            else:
                tmp = t[(t[metric] > threshold)]
            fraction = len(tmp) / float(len(t))
            ax[i].bar(x, fraction, width=0.1, color=colors[c])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0, len(condition_values) * 0.2, 0.2))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_ylabel('fraction of cells')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[1].set_xlabel('image set')
    if title:
        plt.suptitle(title, x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    elif less_than:
        plt.suptitle('fraction of cells with\n' + metric + ' < ' + str(threshold), x=0.63, y=.94,
                     horizontalalignment='center', fontsize=16)
    else:
        plt.suptitle('fraction of cells with\n' + metric + ' > ' + str(threshold), x=0.63, y=.94,
                     horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.35)
    #     plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(hspace=.3)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_' + condition + '_fraction_stacked' + suffix)


def plot_fraction_cells_over_threshold_areas(cell_df, metric, threshold, less_than=False, save_figures=False,
                                             save_dir=None, folder=None):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    image_sets = ut.get_image_sets(df)
    areas = ['VISp', 'VISal']
    colors = ut.get_colors_for_areas(df)
    figsize = (6, 3)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, image_set in enumerate(image_sets):
            if c == 0:
                x = 0
            else:
                x = x + 0.5
            for a, area in enumerate(areas):
                t = df[(df.cre_line == cre_line) & (df.image_set == image_set) & (df.area == area)]
                if less_than:
                    tmp = t[(t[metric] < threshold)]
                else:
                    tmp = t[(t[metric] > threshold)]
                fraction = len(tmp) / float(len(t))
                if a == 0:
                    ax[i].bar(x, fraction, width=0.1, color=colors[a])
                else:
                    ax[i].bar(x + 0.1, fraction, width=0.1, color=colors[a])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0.05, len(image_sets) * 0.5, 0.5))
        # ax[i].set_xlim([-0.5,1.5])
        ax[i].set_xticklabels(image_sets)
    ax[0].set_ylabel('fraction of cells')
    if less_than:
        plt.suptitle('fraction of cells with ' + metric + ' < ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    else:
        plt.suptitle('fraction of cells with ' + metric + ' > ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    ax[i].legend(areas, bbox_to_anchor=(1.9, 1))
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(wspace=.2)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_fraction_areas')
