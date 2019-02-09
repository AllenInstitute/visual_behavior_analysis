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

# formatting
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_palette('deep')


def plot_histogram(values, label, color='k', range=(0, 1), ax=None, offset=False):
    results, edges = np.histogram(values, normed=True, range=range, bins=30)
    binWidth = edges[1] - edges[0]
    if offset:
        ax.bar(edges[:-1]+binWidth, results * binWidth, binWidth, color=color, label=label, alpha=0.5)
    else:
        ax.bar(edges[:-1], results * binWidth, binWidth, color=color, label=label, alpha=0.5)
    return ax


def plot_mean_change_responses(df, vmax=0.3, colorbar=False, ax=None, save_dir=None, folder='figure3', use_events=False):
    if use_events:
        vmax = 0.003
        label = 'mean event magnitude'
        suffix = '_events'
    else:
        vmax = 0.3
        label = 'mean dF/F'
        suffix = ''

    image_set = df.image_set.unique()[0]
    trial_type = df.trial_type.unique()[0]
    cre_line = df.cre_line.unique()[0]
    image_names = np.sort(df.change_image_name.unique())

    cells = []
    for image in image_names:
        tmp = df[(df.change_image_name == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell_specimen_id.values[order])
        cells = cells + cell_ids

    if ax is None:
        figsize = (20, 10)
        fig, ax = plt.subplots(1, len(image_names), figsize=figsize, sharey=True, sharex=True)
        ax = ax.ravel()

    for i, image in enumerate(image_names):
        im_df = df[(df.change_image_name == image)]
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
        sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax[i], cmap='viridis', cbar=colorbar,
                    cbar_kws={'label': label})
        xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=2)
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels([int(x) for x in xticklabels])
        if response_array.shape[0] > 300:
            interval = 100
        else:
            interval = 20
        ax[i].set_yticks(np.arange(0, response_array.shape[0], interval))
        ax[i].set_yticklabels(np.arange(0, response_array.shape[0], interval))
        ax[i].set_xlabel('time after change (s)', fontsize=16)
        ax[i].set_title(image)
        ax[0].set_ylabel('cells')
    plt.suptitle(cre_line, x=0.52, y=1.02)
    plt.gcf().subplots_adjust(top=0.9)
    fig.tight_layout()
    if save_dir:
        save_figure(fig, figsize, save_dir, folder,
                    'change_response_matrix_' + cre_line + '_' + image_set + '_' + trial_type + suffix)


def plot_tuning_curve_heatmap(df, title=None, ax=None, save_dir=None, use_events=False):
    # image_set = df.image_set.unique()[0]
    cre_line = df.cre_line.unique()[0]
    # trial_type = df.trial_type.unique()[0]
    #     detectability = get_detectability()
    #     d = detectability[detectability.image_set==image_set]
    #     order = np.argsort(d.detectability.values)
    #     images = d.image_name.values[order]
    if 'image_name' in df.keys():
        image_name = 'image_name'
        suffix = '_flashes'
    else:
        image_name = 'change_image_name'
        suffix = '_trials'
    images = np.sort(df[image_name].unique())

    if cre_line == 'Vip-IRES-Cre':
        interval = 50
    else:
        interval = 100
    if use_events:
        vmax = 0.03
        label = 'mean event magnitude'
        suffix = suffix+'_events'
    else:
        vmax = 0.3
        label = 'mean dF/F'
        suffix = suffix

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
                     vmin=0, vmax=vmax, robust=True, cbar=True,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": label}, ax=ax)

    if title is None:
        title = 'mean response by image'
    ax.set_title(title, va='bottom', ha='center')
    ax.set_xticklabels(images, rotation=90)
    ax.set_ylabel('cells')
    ax.set_yticks(np.arange(0, response_matrix.shape[0], interval))
    ax.set_yticklabels(np.arange(0, response_matrix.shape[0], interval))
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, 'tuning_curve_heatmaps', title + suffix)


def plot_mean_response_by_repeat_heatmap(df, cre_line, title=None, ax=None, use_events=False, save_figures=False,
                                         save_dir=None, folder=None):
    # repeats = np.arange(1,11,1)
    repeats = np.sort(df.repeat.unique())
    df = df[(df.cre_line==cre_line)&(df.pref_stim==True)&(df.repeat.isin(repeats))]
    image_set = df.image_set.unique()[0]
    tmp = df[df.repeat==1]
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
    ax.set_xticklabels(repeats, rotation=90);
    ax.set_ylabel('cells');
    ax.set_xlabel('repeat')
    ax.set_yticks(np.arange(0, response_matrix.shape[0], interval));
    ax.set_yticklabels(np.arange(0, response_matrix.shape[0], interval));
    if save_dir:
        fig.tight_layout()
        save_figure(fig, figsize, save_dir, folder, 'repeat_response_heatmap_'+cre_line+'_'+image_set+suffix)
    return ax


def plot_flashes_on_trace(ax, trial_type=None, omitted=False, flashes=False, window=[-4,4], alpha=0.15):
    frame_rate = 31.
    stim_duration = .25
    blank_duration = .5
    if flashes:
        change_frame = np.abs(window[0]) * frame_rate
        end_frame = (np.abs(window[0]) + window[1]) * frame_rate
    else:
        change_frame = window[1] * frame_rate
        end_frame = (window[1] + np.abs(window[0])) * frame_rate
    interval = blank_duration + stim_duration
    if omitted:
        array = np.arange((change_frame + interval) * frame_rate, end_frame, interval * frame_rate)
    else:
        array = np.arange(change_frame, end_frame, interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + (stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor='gray', edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if trial_type == 'go':
        alpha = alpha * 3
    else:
        alpha
    array = np.arange(change_frame - ((blank_duration) * frame_rate), 0, -interval * frame_rate)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] - (stim_duration * frame_rate)
        ax.axvspan(amin, amax, facecolor='gray', edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_cell_mean_trace_from_mean_df(cell_data, frame_rate=31., ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlims=[-4, 4],
                    ax=None):
    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    trace = cell_data.mean_trace.values[0]
    times = np.arange(0, len(trace), 1)
    sem = cell_data.sem_trace.values[0]
    ax.plot(trace, label=legend_label, linewidth=3, color=color)
    ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=color)

    xticks, xticklabels = sf.get_xticks_xticklabels(trace, frame_rate, interval_sec, window=xlims)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax

def plot_hist_for_cre_lines(df, metric, range=None, ax=None, save_figures=False, save_dir=None, folder=None, offset=False):
    colors = ut.get_colors_for_cre_lines()
    if range is None:
        range = (0,1)
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
    cre_lines = ut.get_cre_lines(df)
    for i,cre_line in enumerate(cre_lines):
        values = df[(df.cre_line==cre_line)][metric].values
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
        save_figure(fig ,figsize, save_dir, folder, metric+'_across_cre_lines_hist')
    return ax


def plot_session_averages_for_cre_lines(metric, session_summary_df, ax=None, ylims=None, color_by_area=False,
                          save_figures=False, save_dir=None, folder=None):
    if ylims:
        ylims = ylims
    else:
        ylims = (0, 1)
    if ax is None:
        figsize = (6, 5)
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
    figsize=(10,5)
    fig, ax = plt.subplots(1,2,figsize=figsize)
    ax = ax.ravel()
    ax[0] = plot_hist_for_cre_lines(data, metric, range=range, ax=ax[0], save_figures=False, offset=offset)
    ax[1] = plot_session_averages_for_cre_lines(metric, session_summary_df, ax=ax[1], ylims=ylims, color_by_area=color_by_area)
    fig.tight_layout()
    if save_figures:
        save_figure(fig ,figsize, save_dir, folder, metric+'_across_cre_lines')
    return ax


def plot_hist_for_repeats(df, metric, range=None, cre_line=None, ax=None, save_figures=False, save_dir=None, folder=None):
    colors = sns.color_palette()
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    for i,repeat in enumerate(np.sort(df.repeat.unique())):
        values = df[df.repeat==repeat][metric].values
        ax = plot_histogram(values, label='flash number '+str(int(repeat)),
                                color=colors[i], range=range, ax=ax)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    ax.legend()
    if save_figures:
        fig.tight_layout()
        save_figure(fig ,figsize, save_dir, folder, metric+'_by_flash_number_dist_'+cre_line.split('-')[0])
    return ax


def plot_cdf_for_repeats(df, metric, cre_line=None, ax=None, save_figures=False, save_dir=None, folder=None):
    colors = sns.color_palette()
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    for i,repeat in enumerate(np.sort(df.repeat.unique())):
        values = df[df.repeat==repeat][metric].values
        ax = sns.distplot(values[~np.isnan(values)],hist=False, hist_kws={'cumulative':True}, color=colors[i],
                          kde_kws={'cumulative':True}, label=repeat, ax=ax)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    ax.legend()
    if save_figures:
        fig.tight_layout()
        save_figure(fig ,figsize, save_dir, folder, metric+'_by_flash_number_cdf_'+cre_line.split('-')[0])
    return ax


def plot_violin_for_repeats(df, metric, cre_line=None, ax=None, save_figures=False, save_dir=None, folder=None):
    colors = sns.color_palette()
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    ax = sns.violinplot(data=df,x='repeat',y=metric,color='white', ax=ax, cut=0-1)
    ax = sns.stripplot(data=df,x='repeat',y=metric, ax=ax, palette=colors)
    if save_figures:
        save_figure(fig ,figsize, save_dir, folder, metric+'_by_flash_number_violin_'+cre_line.split('-')[0])
    return ax


def plot_distributions_for_repeats(df, metric, cre_line, save_dir=None, folder=None, save_figures=False, ymax=None, range=(0,1)):
    image_set = df.image_set.unique()[0]
    figsize = (15,5)
    fig, ax = plt.subplots(1,3,figsize=figsize)
    ax = ax.ravel()
    ax[0] = plot_hist_for_repeats(df, metric, ax=ax[0], range=range)
    ax[1] = plot_cdf_for_repeats(df, metric, ax=ax[1])
    ax[2] = plot_violin_for_repeats(df, metric, ax=ax[2])
    plt.suptitle(cre_line, fontsize=16, x=0.52, y=1.0)
    fig.tight_layout()
    plt.gcf().subplots_adjust(top=0.9)
    if save_figures:
        save_figure(fig ,figsize, save_dir, folder, metric+'_by_flash_number_'+cre_line.split('-')[0]+'_'+image_set)


def plot_hist_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False):
    colors = ut.get_colors_for_image_sets()
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    for i,image_set in enumerate(np.sort(df.image_set.unique())):
        values = df[df.image_set==image_set][metric].values
        ax = plot_histogram(values, label=image_set, color=colors[i], range=(0,1), ax=ax)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    ax.legend()
#     if save_figures:
#         fig.tight_layout()
#         psf.save_figure(fig ,figsize, save_dir, folder, metric+'_by_flash_number_dist_'+cre_line.split('-')[0])
    return ax


def plot_cdf_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False):
    colors = ut.get_colors_for_image_sets()
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    for i,image_set in enumerate(np.sort(df.image_set.unique())):
        values = df[df.image_set==image_set][metric].values
        ax = sns.distplot(values[~np.isnan(values)],hist=True, hist_kws={'cumulative':True,'histtype':'step'},
                          kde_kws={'cumulative':True}, ax=ax, color=colors[i])
    ax.set_xlabel(metric)
    ax.set_ylabel('cumulative fraction')
    ax.legend()
#     if save_figures:
#         fig.tight_layout()
#         psf.save_figure(fig ,figsize, save_dir, 'figure4', metric+'_by_flash_number_cdf_'+cre_line.split('-')[0])
    return ax


def plot_violin_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False):
    colors = ut.get_colors_for_image_sets()
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    ax = sns.violinplot(data=df,x='image_set',y=metric,color='white', ax=ax, cut=0-1)
    ax = sns.stripplot(data=df,x='image_set',y=metric, ax=ax, palette=colors)
#     if save_figures:
#         psf.save_figure(fig ,figsize, save_dir, 'figure4', metric+'_by_flash_number_violin_'+cre_line.split('-')[0])
    return ax


def plot_session_average_for_image_sets(df, metric, cre_line=None, ax=None, save_figures=False, ymax=None):
    if ax is None:
        figsize = (6,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(cre_line)
    image_sets = ut.get_image_sets(df)
    for i,image_set in enumerate(np.sort(df.image_set.unique())):
        idf = df[(df.image_set==image_set)]
        for experiment_id in idf.experiment_id.unique():
            edf = idf[(idf.experiment_id==experiment_id)]
            area = edf.area.unique()[0]
            color = ut.get_color_for_area(area)
            mean = np.nanmean(edf[metric].values)
            ax.plot(i ,mean, 'o', color=color)
    ax.set_xlabel('image_set')
    ax.set_ylabel(metric)
    ax.set_title('session average')
    ax.set_xticks(np.arange(0,4,1))
    ax.set_xticklabels(image_sets)
    if ymax is None:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim([0,y_max+(0.1*y_max)])
    else:
        ax.set_ylim([0,ymax])
    # make legend just for 2 points
    for area in np.sort(df.area.unique())[::-1]:
        experiment_id = idf[idf.area=='VISp'].experiment_id.unique()[0] #get one experiment for this area
        mean = np.nanmean(df[df.experiment_id==experiment_id][metric].values)
        color = ut.get_color_for_area(area)
        ax.plot(0 ,mean, 'o', color=color, label=area)
    ax.legend()
    #     if save_figures:
    #         fig.tight_layout()
    #         psf.save_figure(fig ,figsize, save_dir, 'figure4', metric+'_by_flash_number_cdf_'+cre_line.split('-')[0])
    return ax


def plot_distributions_for_image_sets(df, metric, cre_line, save_dir=None, folder=None, save_figures=False, ymax=None):
    figsize = (20,5)
    fig, ax = plt.subplots(1,4,figsize=figsize)
    ax = ax.ravel()
    ax[0] = plot_hist_for_image_sets(df, metric, ax=ax[0])
    ax[1] = plot_cdf_for_image_sets(df, metric, ax=ax[1])
    ax[2] = plot_violin_for_image_sets(df, metric, ax=ax[2])
    ax[3] = plot_session_average_for_image_sets(df, metric, ax=ax[3], ymax=ymax)
    plt.suptitle(cre_line, fontsize=16, x=0.52, y=1.0)
    fig.tight_layout()
    plt.gcf().subplots_adjust(top=0.9)
    if save_figures:
        save_figure(fig ,figsize, save_dir, folder, metric+'_by_image_set_'+cre_line.split('-')[0])

