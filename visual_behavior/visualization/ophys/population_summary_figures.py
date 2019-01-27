"""
Created on Thursday January 3 2019

@author: marinag
"""
import numpy as np
import matplotlib.pyplot as plt
import visual_behavior.visualization.ophys.summary_figures as sf
from visual_behavior.visualization.utils import save_figure
import seaborn as sns

# formatting
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_palette('deep')


def plot_histogram(values, label, color='k', range=(0, 1), ax=None):
    results, edges = np.histogram(values, normed=True, range=range, bins=50)
    binWidth = edges[1] - edges[0]
    ax.bar(edges[:-1], results * binWidth, binWidth, color=color, label=label, alpha=0.5)
    return ax


def get_colors_for_cre_lines():
    colors = [sns.color_palette()[2], sns.color_palette()[4]]
    return colors


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
    images = np.sort(df.change_image_name.unique())

    if cre_line == 'Vip-IRES-Cre':
        interval = 50
    else:
        interval = 100
    if use_events:
        vmax = 0.03
        label = 'mean event magnitude'
        suffix = '_events'
    else:
        vmax = 0.3
        label = 'mean dF/F'
        suffix = ''

    cell_list = []
    for image in images:
        tmp = df[(df.change_image_name == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell_specimen_id.values[order])
        cell_list = cell_list + cell_ids

    response_matrix = np.empty((len(cell_list), len(images)))
    for i, cell in enumerate(cell_list):
        responses = []
        for image in images:
            response = df[(df.cell_specimen_id == cell) & (df.change_image_name == image)].mean_response.values[0]
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
