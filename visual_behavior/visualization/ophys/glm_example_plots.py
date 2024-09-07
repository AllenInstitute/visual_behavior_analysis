"""
Created on Wednesday February 23 2022

@author: marinag
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.processing as processing
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.visualization.ophys.platform_paper_figures as ppf

import mindscope_utilities.general_utilities as ms_utils

from visual_behavior_glm.glm import GLM
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_schematic_plots as gsp
import visual_behavior_glm.GLM_visualization_tools as gvt


# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
# sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
sns.set_palette('deep')


def load_glm_model_fit_results(ophys_experiment_id):
    '''
    Load cell_results_df, results, and dropouts from pre-saved files derived from GLM object
    :param ophys_experiment_id:
    :return:
    '''
    # load GLM fit results
    platform_cache_dir = loading.get_platform_analysis_cache_dir()
    fits_dir = os.path.join(platform_cache_dir, 'glm_results', 'model_fits')
    filename = [file for file in os.listdir(fits_dir) if str(ophys_experiment_id) in file and 'cell_results_df.h5' in file]
    cell_results_df = pd.read_hdf(os.path.join(fits_dir, filename[0]), key='df', index_col=0)
    # all results
    filename = [file for file in os.listdir(fits_dir) if str(ophys_experiment_id) in file and 'results_df.h5' in file and 'cell' not in file]
    results = pd.read_hdf(os.path.join(fits_dir, filename[0]), key='df', index_col=0)
    # dropouts
    filename = [file for file in os.listdir(fits_dir) if str(ophys_experiment_id) in file and 'dropouts.pkl' in file]
    dropouts = pd.read_pickle(os.path.join(fits_dir, filename[0]))

    return cell_results_df, results, dropouts


def get_glm_model_fit_cell_results_df(ophys_experiment_id):
    '''
    Load GLM object and extract cell_results_df which contains model fits, then save it
    If cell_results_df is already saved, then load it
    :param ophys_experiment_id
    :return: cell_results_df, table of model fits
    '''
    platform_cache_dir = loading.get_platform_analysis_cache_dir()
    glm_version = '24_events_all_L2_optimize_by_session'
    fit_filepath_start = os.path.join(platform_cache_dir, 'glm_results', 'model_fits', str(ophys_experiment_id))
    fit_filepath = os.path.join(fit_filepath_start + '_cell_results_df.h5')
    print('filepath at', fit_filepath)
    if os.path.exists(fit_filepath):
        print('file exists, loading')
        cell_results_df = pd.read_hdf(fit_filepath, key='df')
    else:
        print('creating GLM object & saving results')
        glm = GLM(ophys_experiment_id, glm_version, log_results=False, log_weights=False, use_previous_fit=True,
                  recompute=False, use_inputs=False, inputs=None, NO_DROPOUTS=False, TESTING=False)
        # cell results df
        cell_results_df = glm.cell_results_df.copy()
        cell_results_df.to_hdf(fit_filepath_start + '_cell_results_df.h5', key='df')
        # results df
        results_df = glm.results.copy()
        results_df.to_hdf(fit_filepath_start + '_results_df.h5', key='df')
        # dropouts
        dropouts = glm.fit['dropouts'].copy()
        pd.to_pickle(dropouts, fit_filepath_start + '_dropouts.pkl')

    return cell_results_df


def plot_glm_model_fit_examples_with_GLM_class(ophys_experiment_id):
    '''
    load GLM object then plot example cells using GLM repo plotting code
    '''
    import visual_behavior_glm.GLM_schematic_plots as gsm
    glm_version = '24_events_all_L2_optimize_by_session'
    # get run params
    platform_cache_dir = loading.get_platform_analysis_cache_dir()
    run_params = pd.read_pickle(os.path.join(platform_cache_dir, 'glm_results', glm_version + '_run_params.pkl'))
    # initialize GLM object
    glm = GLM(ophys_experiment_id, glm_version, log_results=False, log_weights=False, use_previous_fit=True,
              recompute=False, use_inputs=False, inputs=None, NO_DROPOUTS=False, TESTING=False)

    # get time window with an omission and a change
    times = utils.get_start_end_time_for_period_with_omissions_and_change(glm.session.stimulus_presentations.copy(), n_flashes=16)

    # get high SNR traces
    traces = processing.compute_robust_snr_on_dataframe(glm.session.dff_traces)
    if len(traces) < 15:
        n_cells = len(traces)
    else:
        n_cells = 15
    cell_specimen_ids = traces.sort_values(by='robust_snr', ascending=False).index.values[:n_cells]

    # plot examples
    for cell_specimen_id in cell_specimen_ids:
        gsm.plot_glm_example(glm, cell_specimen_id, run_params, times=times, savefig=True)


def get_n_rows_and_ratios_for_stim_window(stim_table, xlim_seconds):
    stim_table = stim_table[(stim_table.start_time >= xlim_seconds[0]) & (stim_table.end_time <= xlim_seconds[1])]
    n_images = len(stim_table[stim_table.image_name != 'omitted'].image_name.unique())
    n_rows = n_images + 7  # there are 6 non image kernels (omissions, hit, miss, lick, run, pupil), plus a top row for stim times

    # set height ratios depending on how many rows there are
    height_ratios = [1.5]  # top row is larger for stim, licks & rewards
    for x in range(n_images + 6):  # 6 non-image kernels
        height_ratios.append(1)

    return n_rows, height_ratios


def plot_kernel_for_feature(cell_weights, feature, kernels, ax):
    # defaults
    change = False
    omitted = False

    # plot kernel
    kernel_weights = cell_weights[feature + '_weights'].values[0]
    t_array = get_t_array_for_kernel(kernels, feature, frame_rate=31)
    ax.plot(t_array, kernel_weights, color='k')

    # plot stim, change, or omission durations
    if (feature == 'hit') or (feature == 'miss'):
        change = True
    if feature == 'omissions':
        omitted = True
    ax = utils.plot_flashes_on_trace(ax, t_array, change, omitted, alpha=0.2)

    ax.set_xlim(t_array[0], t_array[-1])

    return ax



def plot_coding_score_for_feature(cell_dropouts, feature, ax):

    ax.barh(y=0, width=np.abs(cell_dropouts[feature].values), height=0.75)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0,1)

    return ax



def plot_kernels_for_cell(cell_weights, kernels, dataset, xlim_seconds=None, save_dir=None, ax=None):
    """
    Plots kernel weights for an example cell, using the kernels present in a given time window during the session,
    defined by xlim_seconds
    """

    # get stimulus information
    stim_table = dataset.stimulus_presentations.copy()
    stim_table = loading.limit_stimulus_presentations_to_change_detection(stim_table)
    all_images = np.sort(stim_table[stim_table.image_name != 'omitted'].image_name.unique())

    # if no xlims are provided, use the whole session
    if xlim_seconds is None:
        xlim_seconds = [stim_table.start_time.values[0], stim_table.end_time.values[-1]]

    # limit to time window
    stim_table = stim_table[(stim_table.start_time >= xlim_seconds[0]) & (stim_table.end_time <= xlim_seconds[1])]

    # get n_rows and height ratios based on how many images are in this window
    n_rows, height_ratios = get_n_rows_and_ratios_for_stim_window(stim_table, xlim_seconds)

    if ax is None:
        figsize = (2, 0.5*n_rows)
        fig, ax = plt.subplots(n_rows, 1, figsize=figsize, sharex=False, sharey='col', gridspec_kw={'height_ratios': height_ratios})

    # get kernel indices for the images in this window
    images_in_window = []
    for idx in stim_table.index:
        image_name = stim_table.loc[idx]['image_name']
        if (image_name not in images_in_window) and (image_name != 'omitted'):
            images_in_window.append(image_name)

    i = 1
    # plot kernels for images in this window
    for image in images_in_window:
        kernel_image_index = np.where(all_images == image)[0][0]
        image_kernel = 'image' + str(kernel_image_index)
        ax[i] = plot_kernel_for_feature(cell_weights, feature=image_kernel, kernels=kernels, ax=ax[i])
        i += 1

    # plot all other kernels
    ax[i] = plot_kernel_for_feature(cell_weights, feature='omissions', kernels=kernels, ax=ax[i])
    i += 1
    ax[i] = plot_kernel_for_feature(cell_weights, feature='hits', kernels=kernels, ax=ax[i])
    i += 1
    ax[i] = plot_kernel_for_feature(cell_weights, feature='misses', kernels=kernels, ax=ax[i])
    i += 1
    ax[i] = plot_kernel_for_feature(cell_weights, feature='licks', kernels=kernels, ax=ax[i])
    i += 1
    ax[i] = plot_kernel_for_feature(cell_weights, feature='running', kernels=kernels, ax=ax[i])
    i += 1
    ax[i] = plot_kernel_for_feature(cell_weights, feature='pupil', kernels=kernels, ax=ax[i])

    for i in range(n_rows):
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                               labelbottom=False, labeltop=False, labelright=False, labelleft=False)
        ax[i].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                         labelbottom=False, labeltop=False, labelright=False, labelleft=False)
    # ylims = ax[i][0].get_ylim()
    # ax[i][0].set_yticks([np.round(ylim, 2) for ylim in ylims])
    # ax[i][0].set_yticklabels([np.round(ylim, 2) for ylim in ylims], fontsize=10)
    # ax[i][0].set_ylabel('Weight', fontsize=12)
    # ax[i][0].set_xlabel('Time (s)')


    # ax[i].set_xlabel('Weights')

    plt.subplots_adjust(hspace=0, wspace=0.1)

    if save_dir:
        folder = 'coding_score_examples'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time))+'_kernels')
    return ax


def plot_glm_methods_with_example_cells(ophys_experiment_id, cell_specimen_id_1, cell_specimen_id_2,
                                                weights_df, results_pivoted, kernels, save_dir=None):
    '''
    Plot multi-panel figure with:
     kernel support for a window of time during which there is one omission and one change
     kernels for 2 example cells
     coding scores for 2 example cells
     model fit for 2 example cells during same time window as kernel support plot
    '''

    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    # get GLM results
    cell_results_df, expt_results, expt_dropouts = load_glm_model_fit_results(ophys_experiment_id)

    # get start time using function to optimize events in window
    n_flashes = 30
    times = utils.get_start_end_time_for_period_with_omissions_and_change(dataset.stimulus_presentations, n_flashes)
    times = [np.round(time) for time in times]
    xlim_seconds = [times[0], times[-1]]
    start_time = times[0]
    duration_seconds = n_flashes * 0.75

    # limit to time window
    stim_table = loading.limit_stimulus_presentations_to_change_detection(dataset.stimulus_presentations)
    stim_table = stim_table[(stim_table.start_time >= xlim_seconds[0]) & (stim_table.end_time <= xlim_seconds[1])]

    # get n_rows and height ratios based on how many images are in this window
    n_rows, height_ratios = get_n_rows_and_ratios_for_stim_window(stim_table, xlim_seconds)

    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = utils.placeAxesOnGrid(fig, dim=(n_rows, 1), xspan=(0, 0.4), yspan=(0, 0.4), sharex=True, height_ratios=height_ratios)
    ax = plot_behavior_timeseries_and_GLM_kernel_activations(dataset, start_time, duration_seconds, ax=ax)

    # cell 1 data
    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id_1)
    cell_weights = weights_df[weights_df.identifier == identifier]
    cell_dropouts = results_pivoted[results_pivoted.identifier == identifier]

    # model fits - cell 1
    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.4), yspan=(0.65, 0.75), wspace=0.2)
    ax = plot_model_fits_example_cell(cell_specimen_id_1, dataset, cell_results_df, expt_dropouts, expt_results,
                                      kernel=None, n_flashes=n_flashes, include_events=True, times=times, as_panel=True, ax=ax)
    ax.set_ylabel('Cell 1')

    # kernels - cell 1
    ax = utils.placeAxesOnGrid(fig, dim=(n_rows, 1), xspan=(0.5, 0.57), yspan=(0, 0.4), sharex=False, sharey='col', height_ratios=height_ratios, hspace=0.05)
    ax = plot_kernels_for_cell(cell_weights, kernels, dataset, xlim_seconds, ax=ax)
    ax[1].set_title('Cell 1')

    # coding scores - cell 1
    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.5, 0.57), yspan=(0.45, 0.53))
    ax = plot_coding_score_components_for_cell(cell_specimen_id_1, ophys_experiment_id, results_pivoted, ax=ax)
    ax.set_title('')
    ax.set_ylim(0, 1)

    # cell 2 data
    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id_2)
    cell_weights = weights_df[weights_df.identifier == identifier]
    cell_dropouts = results_pivoted[results_pivoted.identifier == identifier]

    # model fits cell 2
    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.4), yspan=(0.8, 0.9))
    ax = plot_model_fits_example_cell(cell_specimen_id_2, dataset, cell_results_df, expt_dropouts, expt_results,
                                      kernel=None, n_flashes=n_flashes, include_events=True, times=times, as_panel=True, ax=ax)
    ax.set_ylabel('Cell 2')

    # kernels - cell 2
    ax = utils.placeAxesOnGrid(fig, dim=(n_rows, 1), xspan=(0.6, 0.67), yspan=(0, 0.4), sharex=False, sharey='col', height_ratios=height_ratios, hspace=0.05)
    ax = plot_kernels_for_cell(cell_weights, kernels, dataset, xlim_seconds, ax=ax)
    ax[1].set_title('Cell 2')

    # coding scores - cell 2
    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.6, 0.67), yspan=(0.45, 0.53))
    ax = plot_coding_score_components_for_cell(cell_specimen_id_2, ophys_experiment_id, results_pivoted, ax=ax)
    ax.set_title('')
    ax.set_ylim(0, 1)
    ax.set_ylabel('')
    ax.set_yticklabels([])


    if save_dir:
        folder = 'glm_methods_cell_examples'
        metadata_string = utils.get_metadata_string(dataset.metadata)
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)) + '_' + str(cell_specimen_id_1) + '_' + str(cell_specimen_id_2))


def plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, results,
                                 kernel=None, include_events=True, include_dff=False,
                                 times=None, n_flashes=16, linewidth=None,
                                 fontsize=8, as_panel=False, save_dir=None, ax=None):
    '''
    For one cell, plot the cell trace, model fits, and model fits with a specific kernel (such as all-images or omissions) removed
    Inputs are attributes of the GLM class in visual_behavior_glm repo, either derived by instantiating the GLM class,
    or by loading cell_results_df and dropouts from pre-saved files for the experiment in the dataset provided.


    :param cell_specimen_id:
    :param dataset: AllenSDK BehaviorOphysExperiment instance
    :param cell_results_df: table of model fits for each cell in the experiment
    :param dropouts: dropout scores table from GLM class or saved files
    :param kernel: kernel to drop for model fit with kernel removed
    :param include_events: if True, plot events, if False, plot dFF
    :param times: tuple of start and end times within the session to plot the trace & fits over
    :param save_dir: directory to save the plot to
    :return:
    '''

    # get model fits for one cell
    fit = cell_results_df[cell_results_df.cell_specimen_id == cell_specimen_id]

    # filter stim presentations to get change detection block with omitted column as bool
    stimulus_presentations = dataset.stimulus_presentations.copy()
    stimulus_presentations = loading.limit_stimulus_presentations_to_change_detection(stimulus_presentations)
    if times is None:
        # get times for plot by finding a time period with an omission and a change
        times = utils.get_start_end_time_for_period_with_omissions_and_change(stimulus_presentations,
                                                                    n_flashes=n_flashes)
        times = [int(time) for time in times]
    xlim_seconds = [times[0], times[-1]]


    # do the plot
    if ax is None:
        figsize = (8, 2.5)
        fig, ax = plt.subplots(figsize=figsize)

    # time to use for extracting the trace in the relvant time window
    time_vec = (fit['fit_trace_timestamps'] > times[0]) & (fit['fit_trace_timestamps'] < times[1])
    cell_index = np.where(cell_results_df.cell_specimen_id.unique() == cell_specimen_id)[0][0]

    ax = utils.add_stim_color_span(dataset, ax, xlim=xlim_seconds, annotate_changes=False,
                                    label_changes=True, label_omissions=True)
    # # add stimulus, change, and omission times to plot
    # stim = stimulus_presentations.query('start_time > @times[0] & start_time < @times[1]')
    # stim = stim[(stim.omitted == False) & (stim.is_change == False)]
    # for index, time in enumerate(stim['start_time'].values):
    #     ax.axvspan(time, time + 0.25, color='gray', alpha=0.75)
    # change = stimulus_presentations.query('start_time > @times[0] & start_time < @times[1]')
    # change = change[change.is_change]
    # for index, time in enumerate(change['start_time'].values):
    #     ax.axvspan(time, time + 0.25, color=gvt.project_colors()['schematic_change'], alpha=.5, edgecolor=None)
    # omission = stimulus_presentations.query('start_time > @times[0] & start_time < @times[1]')
    # omission = omission[omission.omitted]
    # for index, time in enumerate(omission['start_time'].values):
    #     ax.axvline(time, color=gvt.project_colors()['schematic_omission'], linewidth=1.5, linestyle='--')

    style = gsp.get_example_style()
    if linewidth is None:
        linewidth = 2

    suffix = ''
    
    # Plot Filtered event trace
    if include_events:
        suffix = '_events'
        ax.plot(fit['fit_trace_timestamps'][time_vec],
                fit['events'][time_vec], label='events',
                linewidth=linewidth,
                color=sns.color_palette()[0])

    if include_dff:
        # Plot df/f
        suffix = suffix+'_dff'
        # ax2 = ax.twinx()
        ax.plot(fit['fit_trace_timestamps'][time_vec],
                fit['dff'][time_vec], label='dF/F', color=sns.color_palette()[7],
                linewidth=linewidth)
        ax.set_ylabel('dF/F')
        # ax.legend(loc='upper left', fontsize=12)
    # else:
    #     # Plot df/f
    #     suffix = '_dff'
    #     # ax2 = ax.twinx()
    #     ax.plot(fit['fit_trace_timestamps'][time_vec],
    #              fit['dff'][time_vec], label='dF/F', color='gray',
    #              linewidth=linewidth)
    #     ax.set_ylabel('dF/F')
    #     ax.legend(loc='upper left', fontsize=fontsize)

    # Plot Model fits

    # ax.plot(fit['fit_trace_timestamps'][time_vec],
    #         dropouts['Full']['full_model_train_prediction'][time_vec, cell_index],
    #         linewidth=linewidth, color=sns.color_palette()[6], alpha=0)
    ax2 = ax.twinx()
    ax2.plot(fit['fit_trace_timestamps'][time_vec],
             dropouts['Full']['full_model_train_prediction'][time_vec, cell_index],
             label='full model', linewidth=linewidth, color=sns.color_palette()[6])

    if kernel is not None:
        cs = np.round(results.loc[cell_specimen_id]
                      [kernel + '__dropout'] * -1, 3)
        dropout = cs * 100
        ax.plot(fit['fit_trace_timestamps'][time_vec],
                dropouts[kernel]['full_model_train_prediction'][time_vec, cell_index], '-',
                # label='without ' + kernel + ' kernels\n' +
                label = str(np.round(dropout, 1)) + '% reduction in VE\n' + kernel + ' coding score: ' + str(cs),
                linewidth=linewidth, color=sns.color_palette()[2])
        ax.spines['right'].set_visible(False)

    ax.set_xlim(xlim_seconds)
    ax2.set_xlim(xlim_seconds)
    # Clean up plot
    if as_panel:
        # ax.legend(bbox_to_anchor=(1, 1), fontsize=fontsize)
        # ax2.legend(bbox_to_anchor=(1.35, 1), fontsize=fontsize)
        # ax.tick_params(axis='both', which='both', labelsize=8)
        # ax.set_ylabel('Calcium\nevents', fontsize=12)
        # ax.set_xlabel('')

        ax.set_title('')
        # get ylims of data in this window
        ymin, ymax = ax2.get_ylim()
        xmin, xmax = ax2.get_xlim()
        # create scale bar
        ytop = np.round(ymax * .3, 2)
        ax2.set_yticks([0, ytop])
        ax2.set_yticklabels(['', ytop], va='top', ha='left', fontsize=fontsize-2)
        # ax[i].axvline(x=xlim_seconds[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
        # add scale bar
        ax2.annotate('', xy=(xmax + 0.25, 0), xycoords='data', xytext=(xmax + 0.25, ytop),
                       fontsize=fontsize, arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), annotation_clip=False)

        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax2.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.tick_params(which='both', bottom=False, top=False, right=False, left=False,
                       labelbottom=False, labeltop=False, labelright=False, labelleft=True)
        ax2.tick_params(which='both', bottom=False, top=False, right=False, left=False,
                        labelbottom=False, labeltop=False, labelright=True, labelleft=False)
    else:
        # ax.legend(bbox_to_anchor=(1,1), fontsize=fontsize)
        # ax2.legend(loc='upper left', fontsize=fontsize)
        ax.set_ylabel('Calcium events', fontsize=style['fs1'])
        ax.set_xlabel('Time in session (s)', fontsize=style['fs1'])
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', labelsize=style['fs2'])
        ax.tick_params(axis='y', labelsize=style['fs2'])
        ax.set_title('csid: ' + str(cell_specimen_id))
        ax.spines[['bottom', 'left', 'right']].set_visible(False)
        # ax2.spines[['top', 'bottom', 'left']].set_visible(False)
        ax.tick_params(which='both', bottom=True, top=False, right=False, left=True,
                       labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    # ax.set_ylim(-0.035,.9)
    ax.set_xlim(times)

    ax.set_xticks(np.arange(times[0], times[1], 4))


    if save_dir:
        fig.legend(bbox_to_anchor=(1.1, 1), fontsize=fontsize)
        if kernel is not None:
            suffix = suffix + '_' + kernel + '_dropout'
        else:
            suffix = suffix
        m = dataset.metadata.copy()
        filename = str(m['ophys_experiment_id']) + '_' + str(
            cell_specimen_id) + '_' + m['cre_line'].split('-')[0] + '_model_fit' + suffix
        utils.save_figure(fig, figsize, save_dir, 'example_model_fits', filename)

    return ax


def plot_single_cell_example_model_fits_and_behavior(dataset, start_time, duration_seconds, cell_specimen_ids,
                                                     cell_results_df, dropouts, expt_results, fontsize=8,
                                                     kernel='all-images', label_csids=True, linewidth=None,
                                                     short_title=False, include_events=False, save_dir=None, ax=None, suffix=''):
    """
    Plots licking behavior, rewards, running speed, pupil area, and dff traces for a defined window of time.
    Each timeseries gets its own row.
    If no cell_specimen_ids are provided, selects the top 20 cell traces with highest SNR to plot
    if use_filtered_events is True, will plot filtered events trace, if False, will plot dFF
    """

    # xlim_seconds = [start_time - (duration_seconds / 3.), start_time + duration_seconds * 2]
    xlim_seconds = [start_time, start_time + duration_seconds]

    # get behavior timeseries
    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    # licks[:] = -2

    reward_timestamps = dataset.rewards.timestamps.values
    rewards = np.zeros(len(reward_timestamps))
    # rewards[:] = -2

    # get run speed trace and timestamps
    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values
    # limit running trace to window so yaxes scale properly
    start_ind = np.where(running_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(running_timestamps > xlim_seconds[1])[0][0]
    running_speed = running_speed[start_ind:stop_ind]
    running_timestamps = running_timestamps[start_ind:stop_ind]

    # get pupil width trace and timestamps
    eye_tracking = dataset.eye_tracking.copy()
    pupil_diameter = eye_tracking.pupil_width.values
    pupil_diameter[eye_tracking.likely_blink == True] = np.nan
    pupil_timestamps = eye_tracking.timestamps.values
    # smooth pupil diameter
    from scipy.signal import medfilt
    pupil_diameter = medfilt(pupil_diameter, kernel_size=5)
    # limit pupil trace to window so yaxes scale properly
    start_ind = np.where(pupil_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(pupil_timestamps > xlim_seconds[1])[0][0]
    pupil_diameter = pupil_diameter[start_ind:stop_ind]
    pupil_timestamps = pupil_timestamps[start_ind:stop_ind]

    # make the plot
    if ax is None:
        figsize = (8, 6)
        fig, ax = plt.subplots(len(cell_specimen_ids)+3, 1, figsize=figsize, sharex=True, )
        ax = ax.ravel()


    for i, cell_specimen_id in enumerate(cell_specimen_ids):

        ax[i] = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                            kernel=kernel, times=xlim_seconds, linewidth=1.5, fontsize=fontsize,
                                            include_events=include_events, include_dff=True, as_panel=True, ax=ax[i])

        ax[i].set_xlim(xlim_seconds)
        # ax[0].legend(bbox_to_anchor=(1, 1), fontsize=fontsize-2)
        # if i == 0:
        #     ax[i].legend(loc='upper left', fontsize=fontsize-2)
        # else:
        #     ax[i].get_legend().remove()

        # label cell id on right side so it can be cropped out later
        xmin, xmax = ax[i].get_xlim()
        if label_csids:
            ax[i].text(s='csid:' + str(cell_specimen_id), x=xmax + 0.2, y=0, fontsize=fontsize-2)

        # get ylims of data in this window
        ymin, ymax = ax[i].get_ylim()
        # create scale bar
        ax[i].set_yticks([0, np.round(ymax * .3, 2)])
        ax[i].set_yticklabels(['', np.round(ymax * .3, 2)], va='top', ha='right', fontsize=fontsize-2)
        # ax[i].axvline(x=xlim_seconds[0] - 0.1, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
        # add scale bar
        ax[i].annotate('', xy=(xmin-0.2, 0), xycoords='data', xytext=(xmin-0.2, np.round(ymax * 0.3, 2)), fontsize=fontsize,
                       arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), annotation_clip=False)

        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        # if i < len(cell_specimen_ids) - 1:
        ax[i].set_xticklabels([])
        ax[i].set_xlabel('')
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)

        # plot cell ID on left with index
        ax[i].set_ylabel('cell ' + str(i + 1), rotation=0, fontsize=fontsize, ha='center', y=0.6)

    # plot behavior timeseries
    colors = sns.color_palette()

    ax[i+2].plot(running_timestamps, running_speed, label='running_speed', color='gray', linewidth=linewidth)
    ax[i+2].set_ylabel('running\nspeed\n(cm/s)', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax[i+2].set_ylim(ymin=-8)

    ax[i+3].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color='gray', linewidth=linewidth)
    ax[i+3].set_ylabel('pupil\ndiameter\n(pixels)', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)

    ax[i+1].plot(lick_timestamps, licks, '|', label='licks', color='gray', markersize=10)
    ax[i+1].plot(reward_timestamps, rewards, 'o', label='rewards', color='cyan', markersize=5)
    ax[i+1].set_ylabel('licks &\nrewards', rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=fontsize)
    ax[i+1].set_yticklabels([])
    ax[i+1].set_ylim(-1,3)

    # add stimuli, despine and such
    for x in range(i+1, i+4):
        ax[x] = ppf.add_stim_color_span(dataset, ax[x], xlim=xlim_seconds, label_changes=True, label_omissions=True)
        sns.despine(ax=ax[x], top=True, right=True, left=True, bottom=True)
        ax[x].set_xticklabels([])
        ax[x].set_xlabel('')
        ax[x].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True, labelsize=fontsize)

    # Label bottom row with times
    ax[i+3].set_xticks(np.arange(int(xlim_seconds[0]), int(xlim_seconds[1]), 10))
    ax[i+3].set_xticklabels(np.arange(int(xlim_seconds[0]), int(xlim_seconds[1]), 10), fontsize=fontsize)
    # label bottom row of plot
    ax[i+3].set_xlabel('Time in session (seconds)', fontsize=fontsize+2)
    ax[i+3].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True, labelsize=fontsize)

    # add title to top row
    # ophys_container_id = dataset.metadata['ophys_container_id']
    # metadata_string = str(ophys_container_id)+'_'+utils.get_metadata_string(dataset.metadata) +'_'+ str(int(start_time)) +'_'+ str(duration_seconds)
    m = dataset.metadata.copy()
    metadata_string = str(m['ophys_container_id']) + '_' + str(m['mouse_id']) + '_' + m['cre_line'].split('-')[
        0] + '_' + str(m['targeted_structure']) + '_' + str(m['imaging_depth']) + '_' + m['session_type']  +'_'+ str(int(start_time)) +'_'+ str(duration_seconds)

    if short_title:
        metadata_short = str(m['ophys_container_id']) + '_' + m['session_type']  +'_'+ str(int(start_time)) +'_'+ str(duration_seconds)
        # ax[0].set_title(metadata_short, fontsize=6, va='top', )
        ax[0].set_title('')
    else:
        plt.suptitle(metadata_string, x=0.5, y=0.97, fontsize=14)
    plt.subplots_adjust(hspace=0)
    if save_dir:
        print('saving')
        folder = 'example_cell_model_fits_and_behavior_stacked'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + suffix)
    return ax



def plot_single_cell_model_fit_and_kernels(dataset, matched_cells, start_time, duration_seconds,
                                           cell_results_df, dropouts, expt_results, run_params,
                                           weights_df, results_pivoted, fontsize=10, include_events=True,
                                           save_dir=None):


    figsize = [12, 4]
    fig = plt.figure(figsize=figsize, facecolor='white')

    kernels = run_params['kernels']
    features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'omissions', 'hits', 'misses', 'running', 'pupil', 'licks', ]
    widths = [1] + [kernels[feature]['length'] for feature in features[8:]]

    x = 0
    ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells) + 3, 1),
                                xspan=(x, x + 0.4), yspan=(0, 1), sharex=True, wspace=0.5)

    ax = plot_single_cell_example_model_fits_and_behavior(dataset, start_time, duration_seconds, matched_cells,
                                                        cell_results_df, dropouts, expt_results,
                                                        kernel=None, label_csids=False, linewidth=None,
                                                        fontsize=fontsize, short_title=True, include_events=include_events,
                                                        save_dir=None, ax=ax, suffix='')
    ax[0].set_title('Model fits', fontsize=fontsize)

    x = 0.5
    ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells) +3, 7),
                                xspan=(x, x + 0.38), yspan=(0, 1), sharey=True, sharex=True, wspace=0.2,
                                width_ratios=widths)

    for i, cell_specimen_id in enumerate(matched_cells):
        if i == len(matched_cells)-1:
            label_x = True
        else:
            label_x = False
        ax[i] = plot_all_kernels_for_cell(cell_specimen_id, dataset, weights_df, kernels,
                                            fontsize=fontsize, label_x=label_x, ax=ax[i])
        for e in range(7):
            if i!= len(matched_cells)-1:
                ax[i][e].set_xlabel('')
            ax[i][e].set_ylabel('')
            # if i != 0:
            ax[i][e].set_title('')

        # make remaining rows blank
        for i in range(len(matched_cells), len(matched_cells)+3):
            for e in range(7):
                ax[i][e].axis('off')
    ax[0][4].set_title('Kernels', ha='right', fontsize=fontsize)

    ax = utils.placeAxesOnGrid(fig, dim=(len(matched_cells) + 3, 1), xspan=(0.92, 1), yspan=(0, 1), sharex=False, sharey=True)
    for i, cell_specimen_id in enumerate(matched_cells):
        ax[i] = plot_coding_score_components_for_cell(cell_specimen_id, dataset.ophys_experiment_id,
                                                      results_pivoted, fontsize=fontsize-2, as_panel=True, ax=ax[i])
        ax[i].set_ylabel('')
        ax[i].set_ylim(0,1)
        # ax[i].set_xlabel('')
        ax[i].set_xticklabels('')


    feature_labels = ['images', 'omissions', 'task', 'behavior']
    ax[i].set_xticklabels(feature_labels)

    ax[0]. set_title('Coding scores', fontsize=fontsize)

    # make remaining rows blank
    for i in range(len(matched_cells), len(matched_cells) + 3):
        ax[i].axis('off')


    metadata_string = utils.get_metadata_string(dataset.metadata)
    plt.suptitle(metadata_string, x=0.5, y=0.99, fontsize=14)
    if save_dir:
        print('saving')
        suffix = '_model_fits_and_kernels'
        folder = 'example_cell_model_fits_and_behavior_stacked'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + suffix)


def plot_glm_model_fits_examples_for_experiment(ophys_experiment_id, save_dir=None):
    '''
    Loads dataset object & GLM model fits and plots the cell trace, model fit, and model fit with images or omissions removed
    for the 10 cells with highest SNR in the provided experiment
    '''

    # load dataset & filter stim presentations
    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    # get GLM results
    cell_results_df, expt_results, dropouts = load_glm_model_fit_results(ophys_experiment_id)

    # get high SNR traces
    # traces = processing.compute_robust_snr_on_dataframe(dataset.dff_traces)
    # if len(traces) < 10:
    #     n_cells = len(traces)
    # else:
    #     n_cells = 10
    # cell_specimen_ids = traces.sort_values(by='robust_snr', ascending=False).index.values[:n_cells]

    # get high variance explained cells
    cell_specimen_ids = expt_results[(expt_results['Full__avg_cv_adjvar_test_raw'] > 0.1)].index.values

    for cell_specimen_id in cell_specimen_ids:
        # cell_specimen_id = cell_specimen_ids[0]

        plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                     None, include_events=True, save_dir=save_dir)
        plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                     'all-images', include_events=True, save_dir=save_dir)
        plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                     'omissions', include_events=True, save_dir=save_dir)
        plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                     None, include_events=False, save_dir=save_dir)
        plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                     'all-images', include_events=False, save_dir=save_dir)
        plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                     'omissions', include_events=False, save_dir=save_dir)



####### old loading function #########

def load_GLM_outputs(glm_version, experiments_table, cells_table, glm_output_dir=None):
    """
    loads results_pivoted and weights_df from files in base_dir, or generates them from mongo and save to base_dir
    results_pivoted and weights_df will be limited to the ophys_experiment_ids and cell_specimen_ids present in experiments_table and cells_table
    because this function simply loads the results, any pre-processing applied to results_pivoted (such as across session normalization or signed weights) will be used here

    :param glm_version: example = '24_events_all_L2_optimize_by_session'
    :param glm_output_dir: directory containing GLM output files to load and save processed data files to for this iteration of the analysis
                            if None, GLM results will be obtained from mongo and will not be saved out
    :param experiments_table: SDK ophys_experiment table limited to experiments intended for analysis
    :param cells_table: SDK ophys_cell_table limited to cell_specimen_ids intended for analysis
    :return:
        results_pivoted: table of dropout scores for all cell_specimen_ids in cells_table
        weights_df: table with model weights for all cell_specimen_ids in cells_table
    """
    # get GLM kernels and params for this version of the model
    run_params = glm_params.load_run_json(glm_version)
    kernels = run_params['kernels']
    # if glm_output_dir is not None:
    # load GLM results for all cells and sessions from file if it exists otherwise load from mongo
    glm_results_path = os.path.join(glm_output_dir, glm_version + '_results_pivoted.h5')
    if os.path.exists(glm_results_path):
        results_pivoted = pd.read_hdf(glm_results_path, key='df')
    else:
        print('no results_pivoted at', glm_results_path)
        print('please generate before running single cell plots')

    weights_path = os.path.join(glm_output_dir, glm_version + '_weights_df.h5')
    if os.path.exists(weights_path):  # if it exists, load it
        weights_df = pd.read_hdf(weights_path, key='df')
    else:
        print('no weights at', weights_path)
        print('please generate before running single cell plots')

    return results_pivoted, weights_df, kernels


####### plot GLM kernel timing along with behavior timeseries & stimulus times #######

def plot_kernel_activations(dataset, start_time, duration_seconds, kernel='omissions', ax=None):
    '''
    For a given period of time in an ophys or behavior session,
    then plot the window over which a given GLM kernel is active, relative to the event of interest
    for example, omissions kernel is 3 seconds after omission onset, licks kernel is +/-1 second around each lick

    :param dataset: SDK dataset object
    :param start_time: start time for plot (in seconds within session)
    :param duration_seconds: duration in seconds for the x axis to span
    :param kernel: name of kernel to plot time window for, can be ['imagex', 'omissions', 'hits', 'misses', 'licks']
    :param ax:
    :return:
    '''
    if ax is None:
        fig, ax = plt.subplots()

    # xlim_seconds = [start_time - (duration_seconds / 4.), start_time + duration_seconds * 2]
    xlim_seconds = [start_time, start_time + duration_seconds]

    stim_table = dataset.stimulus_presentations.copy()
    stim_table = loading.limit_stimulus_presentations_to_change_detection(stim_table)
    trials = dataset.trials.copy()
    # get all images & assign colors (image colors wont be used if a color is provided or if label_changes is True)
    images = np.sort(stim_table[stim_table.omitted == False].image_name.unique())
    image_colors = sns.color_palette("hls", len(images))
    # limit to time window
    stim_table = stim_table[(stim_table.start_time >= xlim_seconds[0]) & (stim_table.end_time <= xlim_seconds[1])]
    images = stim_table[stim_table.image_name != 'omitted'].image_name.unique()

    # get timestamps & frames for this window
    # get first and last indices
    first_idx = stim_table[stim_table.start_time >= xlim_seconds[0]].index.values[0]
    first_frame = int(stim_table.loc[first_idx].start_frame)
    last_idx = stim_table[stim_table.start_time <= xlim_seconds[1]].index.values[-1]
    last_frame = int(stim_table.loc[last_idx].end_frame)

    # use running timestamps to construct array for plotting kernels in this time window
    running_timestamps = dataset.running_speed.timestamps.values
    kernel_trace_timestamps = running_timestamps[first_frame:last_frame]
    kernel_trace = np.zeros(len(kernel_trace_timestamps))

    # loop through stimulus presentations and add a span with appropriate color
    image_presentation_number = 0
    for idx in stim_table.index:
        start_time = stim_table.loc[idx]['start_time']
        image_name = stim_table.loc[idx]['image_name']
        # image_index = stim_table.loc[idx]['image_index']
        if ('image' in kernel):
            image_kernel_index = int(kernel[-1]) - 1
            if image_name == images[image_kernel_index]:
                # frame where image starts
                frame_in_kernel_trace = stim_table.loc[idx]['start_frame'] - first_frame
                # length of this kernel
                len_image_kernel = 0.75
                # set values during this kernel window to 1 and plot
                kernel_trace[frame_in_kernel_trace:frame_in_kernel_trace + (int(len_image_kernel * 60))] = 1
                # plot first image as black, rest as gray
                if image_presentation_number == 0:
                    ax.plot(kernel_trace_timestamps, kernel_trace, color='k', zorder=10000)
                else:
                    ax.plot(kernel_trace_timestamps, kernel_trace, color='gray', linewidth=0.5)
                image_presentation_number += 1
        if (image_name == 'omitted') and (kernel == 'omissions'):
            # frame where omission starts
            frame_in_kernel_trace = stim_table.loc[idx]['start_frame'] - first_frame
            # length of this kernel
            len_omission_kernel = 3
            # set values during this kernel window to 1 and plot
            kernel_trace[frame_in_kernel_trace:frame_in_kernel_trace + (len_omission_kernel * 60)] = 1
            ax.plot(kernel_trace_timestamps, kernel_trace, color='k')
        if stim_table.loc[idx]['is_change']:
            # index into trials table to see if this change was a hit or not
            is_hit = trials[trials.change_frame == stim_table.loc[idx].start_frame].hit.values[0]
            # get task kernel length
            len_task_kernel = 2.25
            if (kernel == 'hits') and (is_hit == True):
                # get first frame where this kernel starts
                frame_in_kernel_trace = stim_table.loc[idx]['start_frame'] - first_frame
                kernel_trace[frame_in_kernel_trace:frame_in_kernel_trace + (int(len_task_kernel * 60))] = 1
                ax.plot(kernel_trace_timestamps, kernel_trace, color='k')
            elif (kernel == 'misses') and (is_hit == False):
                # get first frame where this kernel starts
                frame_in_kernel_trace = stim_table.loc[idx]['start_frame'] - first_frame
                kernel_trace[frame_in_kernel_trace:frame_in_kernel_trace + (int(len_task_kernel * 60))] = 1
                ax.plot(kernel_trace_timestamps, kernel_trace, color='k')
    if kernel == 'licks':
        licks = dataset.licks.copy()
        lick_frames = licks[(licks.frame >= first_frame) & (licks.frame <= last_frame)].frame.values
        for i, lick_frame in enumerate(lick_frames):
            kernel_trace = np.zeros(len(kernel_trace_timestamps))
            # lick kernel is +/-1 second after each lick
            kernel_trace[int(lick_frame - first_frame) - 60:int(lick_frame - first_frame) + 60] = 1
            if i == 1:
                ax.plot(kernel_trace_timestamps, kernel_trace, color='k', zorder=10000)
            else:
                ax.plot(kernel_trace_timestamps, kernel_trace, color='gray', linewidth=0.5)

    # label y axis with kernel name
    ax.set_ylabel(kernel, rotation=0, horizontalalignment='left', verticalalignment='center')

    return ax


def plot_behavior_timeseries_and_GLM_kernel_activations(dataset, start_time, duration_seconds=20,
                                                        label_stim_times_on_all_rows=False,
                                                        save_dir=None, ax=None):
    """
    For a given period of time in an ophys or behavior session,
    plot the stimulus times, licks and rewards on the top row,
    then plot the window over which a given GLM kernel is active, relative to the event of interest
    for example, omissions kernel is 3 seconds after omission onset, licks kernel is +/-1 second around each lick
    for continuous variables (running and pupil), plot the full timeseries

    if label_stim_times_on_all_rows is True, stimulus times will be shown for all rows

    Selects the top 6 cell traces with highest SNR to plot
    """
    # label_changes = True
    # label_omissions = True
    # start_time = 628
    # duration_seconds = 20

    # get limits for this time window
    # xlim_seconds = [start_time - (duration_seconds / 4.), start_time + duration_seconds * 2]
    xlim_seconds = [start_time, start_time + duration_seconds]

    stim_table = loading.limit_stimulus_presentations_to_change_detection(dataset.stimulus_presentations)

    # get kernel indices for the images in this window
    stim_table = stim_table[(stim_table.start_time >= xlim_seconds[0]) & (stim_table.end_time <= xlim_seconds[1])]
    images_in_window = []
    for idx in stim_table.index:
        image_name = stim_table.loc[idx]['image_name']
        if (image_name not in images_in_window) and (image_name != 'omitted'):
            images_in_window.append(image_name)

    # get number of rows and height ratios depending on how many images are in this window
    n_rows, height_ratios = get_n_rows_and_ratios_for_stim_window(stim_table, xlim_seconds)

    # get behavior events & timeseries to plot
    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    licks[:] = -1

    reward_timestamps = dataset.rewards.timestamps.values
    rewards = np.zeros(len(reward_timestamps))
    rewards[:] = -2

    # get run speed trace and timestamps
    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values
    # limit running trace to window so yaxes scale properly
    start_ind = np.where(running_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(running_timestamps > xlim_seconds[1])[0][0]
    running_speed = running_speed[start_ind:stop_ind]
    running_timestamps = running_timestamps[start_ind:stop_ind]

    # get pupil width trace and timestamps
    eye_tracking = dataset.eye_tracking.copy()
    pupil_diameter = eye_tracking.pupil_width.values
    pupil_diameter[eye_tracking.likely_blink == True] = np.nan
    pupil_timestamps = eye_tracking.timestamps.values
    # smooth pupil diameter
    from scipy.signal import medfilt
    pupil_diameter = medfilt(pupil_diameter, kernel_size=5)
    # limit pupil trace to window so yaxes scale properly
    start_ind = np.where(pupil_timestamps < xlim_seconds[0])[0][-1]
    stop_ind = np.where(pupil_timestamps > xlim_seconds[1])[0][0]
    pupil_diameter = pupil_diameter[start_ind:stop_ind]
    pupil_timestamps = pupil_timestamps[start_ind:stop_ind]

    if ax is None:
        figsize = (10, 6)
        fig, ax = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios':height_ratios})
        ax = ax.ravel()

    colors = sns.color_palette()

    # first axis is licking behavior + stim timing
    ax[0] = ppf.add_stim_color_span(dataset, ax[0], xlim=xlim_seconds)
    ax[0].plot(lick_timestamps, licks, '|', label='licks', color='gray', markersize=10)
    ax[0].plot(reward_timestamps, rewards, '^', label='rewards', color='b', markersize=8)
    ax[0].set_yticklabels([])
    ax[0].set_ylim(-3, 0)
    # ax[0].legend(bbox_to_anchor=(0, 0.529))
    ax[0].legend(loc='upper right', fontsize='xx-small') #bbox_to_anchor=(0.0, 0.6), fontsize='x-small')
    # ax[0].set_ylabel('stimulus & behavior events')

    # next axes are the kernel timings
    i = 1
    for image in images_in_window:
        ax[i] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='image '+str(i), ax=ax[i])
        i+= 1

    ax[i] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='omissions', ax=ax[i])
    i+=1

    ax[i] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='hits', ax=ax[i])
    i+=1

    ax[i] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='misses', ax=ax[i])
    i+=1

    ax[i] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='licks', ax=ax[i])
    i+=1

    for x in range(1, i):
        ax[x].set_ylim(-0.2, 1.2)

    # now plot running and pupil as continuous variables
    ax[i].plot(running_timestamps, running_speed, label='running_speed', color='k')
    ax[i].set_ylabel('running', rotation=0, horizontalalignment='left', verticalalignment='center')
    ax[i].yaxis.set_label_position("right")
    ax[i].set_ylim(ymin=-8)
    i+=1

    ax[i].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color='k')
    ax[i].set_ylabel('pupil', rotation=0, horizontalalignment='left',  verticalalignment='center')
    ax[i].yaxis.set_label_position("right")
    ax[i].set_xticks(np.arange(xlim_seconds[0], xlim_seconds[1], 4))
    i+=1

    # remove ticks everywhere but bottom row + set xlim for all axes
    i = n_rows - 1
    ax[i].set_xlabel('Time in session (seconds)')
    for i in range(n_rows):
        ax[i].set_xlim(xlim_seconds)
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)
        ax[i].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax[i].set_yticklabels([])
        # flip all ylabels to the right side
        ax[i].yaxis.set_label_position("right")
        if label_stim_times_on_all_rows:
            ax[i] = ppf.add_stim_color_span(dataset, ax[i], xlim=xlim_seconds)
    # bottom row has ticks for timestamps
    ax[n_rows - 1].tick_params(which='both', bottom=True, top=False, right=False, left=False,
                               labelbottom=True, labeltop=False, labelright=False, labelleft=True)

    # add title to top row
    metadata_string = utils.get_metadata_string(dataset.metadata)

    ax[0].set_title(metadata_string + '\nEncoding model features')
    # plt.suptitle(metadata_string + '\nEncoding model features', x=0.5, y=1.05, fontsize=18)
    # ax[0].set_title('Encoding model features'
    plt.subplots_adjust(hspace=0, wspace=0.9)

    if save_dir:
        folder = 'behavior_physio_timeseries_GLM_kernels'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)))
    return ax


####### single cell example plots #######

def get_time_window_for_kernel(kernels, feature):
    time_window = (kernels[feature]['offset'], kernels[feature]['offset'] + kernels[feature]['length'])
    return time_window


def get_t_array_for_kernel(kernels, feature, frame_rate):
    time_window = get_time_window_for_kernel(kernels, feature)
    t_array = ms_utils.get_time_array(t_start=time_window[0], t_end=time_window[1], sampling_rate=frame_rate,
                                      include_endpoint=False)
    if 'image' in feature:
        t_array = t_array[:-1]  # not sure why we have to do this
    return t_array


def plot_cell_rois_and_GLM_weights(cell_specimen_id, cells_table, experiments_table, dropout_features, results_pivoted, weights_df,
                                   weights_features, kernels, save_dir=None, folder=None, data_type='dff'):
    """
    This function limits inputs just to the provided cell_specimen_id to hand off to the function plot_matched_roi_and_traces_example_GLM
    That function will plot the following panels:
        cell ROI masks matched across sessions for a given cell_specimen_id,
        change and omission triggered average respones across sessions,
        image locked running and pupil if included 'running' and 'pupil' in included weights_features
        dropout scores across dropout_features and sessions as a heatmap,
        kernels weights across sessions for the kernels in weights_features
    :param cell_specimen_id: cell_specimen_id for cell to plot
    :param cells_table: must only include a max of one one experiment per experience level for a given container. ok if less than 1.
    :param experiments_table: must only include one experiment per experience level for a given container
    :param results_pivoted: must only include one experiment per experience level for a given container
                            must be limited to features for plotting, plus cell_specimen_id + experiment_id
    :param weights_df: must only include one experiment per experience level for a given container
    :param weights_features: columns in weights_df to use for plotting
    :param save_dir: top level directory where files exist for this run of analysis
                        code will create a folder within save_dir called 'matched_cell_examples'
    :param subfolder: name of subfolder to create within os.path.join(save_dir, 'matched_cell_examples') to save plots, ex: 'cluster_0' or 'without_exp_var_full_model' or 'with_running_and_pupil'
    :param data_type: can be 'dff', 'events', or 'filtered_events' - to be used for cell response plots
    :return:
    """

    # make sure weights and dropouts are limited to matched experiments / cells
    cells_table = loading.get_cell_table(platform_paper_only=True, limit_to_closest_active=True,
                                         limit_to_matched_cells=True, add_extra_columns=True)
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=True, limit_to_closest_active=True)
    matched_cells = cells_table.cell_specimen_id.unique()
    matched_experiments = cells_table.ophys_experiment_id.unique()
    weights_df = weights_df[weights_df.ophys_experiment_id.isin(matched_experiments)]
    weights_df = weights_df[weights_df.cell_specimen_id.isin(matched_cells)]
    results_pivoted = results_pivoted.reset_index()  # reset just in case
    results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]

    # get cell info
    cell_metadata = cells_table[cells_table.cell_specimen_id == cell_specimen_id]

    # get metadata for this cell
    cell_metadata = cells_table[cells_table.cell_specimen_id == cell_specimen_id]
    # get weights for example cell
    cell_weights = weights_df[weights_df.cell_specimen_id == cell_specimen_id]
    # if exp var full model is in features (must be first feature), scale it by 10x so its on similar scale as dropouts
    if 'variance_explained_full' in results_pivoted.keys():
        results_pivoted['variance_explained_full'] = results_pivoted['variance_explained_full'] * 10
    # get dropouts just for one cell
    cell_dropouts = results_pivoted[results_pivoted.cell_specimen_id == cell_specimen_id]

    plot_matched_roi_and_traces_example_GLM(cell_metadata, cell_dropouts, cell_weights, weights_features, kernels,
                                            dropout_features, experiments_table, data_type, save_dir, folder)


def plot_matched_roi_and_traces_example_GLM(cell_metadata, cell_dropouts, cell_weights, weights_features, kernels,
                                            dropout_features, experiments_table, data_type, save_dir=None, folder=None):
    """
    This function will plot the following panels:
        cell ROI masks matched across sessions for a given cell_specimen_id,
        change and omission triggered average respones across sessions,
        image locked running and pupil if included 'running' and 'pupil' in included weights_features,
        dropout scores across features and sessions as a heatmap,
        kernels weights across sessions for the kernels in weights_features.
    Plots the ROI masks and cell traces for a cell matched across sessions, along with dropout scores and weights for images, hits, misses and omissions
    Cell_metadata is a subset of the ophys_cells_table limited to the cell_specimen_id of interest
    cell_dropouts is a subset of the results_pivoted version of GLM output limited to cell_specimen_id of interest
    cell_weights is a subset of the weights matrix from GLM limited to cell_specimen_id of interest
    all input dataframes must be limited to last familiar and second novel active (i.e. max of one session per type)
    if one session type is missing, the max projection but no ROI will be plotted and the traces and weights will be missing for that experience level
    """

    if len(cell_metadata.cell_specimen_id.unique()) > 1:
        print('There is more than one cell_specimen_id in the provided cell_metadata table')
        print('Please limit input to a single cell_specimen_id')

    # set up plotting for each experience level
    experience_levels = ['Familiar', 'Novel 1', 'Novel >1']
    colors = utils.get_experience_level_colors()
    n_exp_levels = len(experience_levels)
    # get relevant info for this cell
    cell_metadata = cell_metadata.sort_values(by='experience_level')
    cell_specimen_id = cell_metadata.cell_specimen_id.unique()[0]
    ophys_container_id = cell_metadata.ophys_container_id.unique()[0]
    # need to get all experiments for this container, not just for this cell
    ophys_experiment_ids = experiments_table[experiments_table.ophys_container_id == ophys_container_id].index.values
    n_expts = len(ophys_experiment_ids)
    if n_expts > 3:
        print('There are more than 3 experiments for this cell. There should be a max of 1 experiment per experience level')
        print('Please limit input to only one experiment per experience level')

    # set up labels for different trace types
    if data_type == 'dff':
        ylabel = 'dF/F'
    else:
        ylabel = 'response'

    # number of columns is one for each experience level,
    # plus additional columns for stimulus and omission traces, and running and pupil averages (TBD)
    extra_cols = 2
    if 'running' in weights_features:
        extra_cols += 1
    if 'running' in weights_features:
        extra_cols += 1
    n_cols = n_exp_levels + extra_cols
    print(extra_cols, 'extra cols')

    figsize = (3.5 * n_cols, 6)
    fig, ax = plt.subplots(2, n_cols, figsize=figsize)
    ax = ax.ravel()

    print('cell_specimen_id:', cell_specimen_id)
    # loop through experience levels for this cell
    for e, experience_level in enumerate(experience_levels):
        print('experience_level:', experience_level)

        # get ophys_experiment_id for this experience level
        # experiments_table must only include one experiment per experience level for a given container
        ophys_experiment_id = experiments_table[(experiments_table.ophys_container_id == ophys_container_id) &
                                                (experiments_table.experience_level == experience_level)].index.values[0]
        print('ophys_experiment_id:', ophys_experiment_id)
        ind = experience_levels.index(experience_level)
        color = colors[ind]

        # load dataset for this experiment
        dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)

        try:  # attempt to generate plots for this cell in this this experience level. if cell does not have this exp level, skip
            # plot ROI mask for this experiment
            ct = dataset.cell_specimen_table.copy()
            cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id  # typically will fail here if the cell_specimen_id isnt in the session
            roi_masks = dataset.roi_masks.copy()  # save this to get approx ROI position if subsequent session is missing the ROI (fails if the first session is the one missing the ROI)
            ax[e] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                      spacex=50, spacey=50, show_mask=True, ax=ax[e])
            ax[e].set_title(experience_level, color=color)

            # get change responses and plot on second to next axis after ROIs (there are n_expts # of ROIs)
            window = [-1, 1.5]  # window around event
            sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                                   data_type=data_type, event_type='changes', load_from_file=True)
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == True)]

            ax[n_expts] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                ylabel=ylabel, legend_label=None, color=color, interval_sec=1,
                                                xlim_seconds=window, plot_sem=True, ax=ax[n_expts])
            ax[n_expts] = utils.plot_flashes_on_trace(ax[n_expts], cell_data.trace_timestamps.values[0],
                                                      change=True, omitted=False)
            ax[n_expts].set_title('changes')

            # get omission responses and plot on last axis
            sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                                   data_type=data_type, event_type='omissions', load_from_file=True)
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.omitted == True)]

            ax[n_expts + 1] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                    ylabel=ylabel, legend_label=None, color=color, interval_sec=1,
                                                    xlim_seconds=window, plot_sem=True, ax=ax[n_expts + 1])
            ax[n_expts + 1] = utils.plot_flashes_on_trace(ax[n_expts + 1], cell_data.trace_timestamps.values[0],
                                                          change=False, omitted=True)
            ax[n_expts + 1].set_title('omissions')

            if 'running' in weights_features:
                pass
            if 'pupil' in weights_features:
                pass

        except BaseException:  # plot area of max projection where ROI would have been if it was in this session
            # plot the max projection image with the xy location of the previous ROI
            # this will fail if the familiar session is the one without the cell matched
            print('no cell ROI for', experience_level)
            ax[e] = sf.plot_cell_zoom(roi_masks, dataset.max_projection, cell_roi_id,
                                      spacex=50, spacey=50, show_mask=False, ax=ax[e])
            ax[e].set_title(experience_level)

        # try: # try plotting GLM outputs for this experience level
        if 'running' in weights_features:
            pass
        if 'pupil' in weights_features:
            pass

        # GLM plots start after n_expts for each ROI mask, plus n_extra_cols more axes for omission and change responses (and running and pupil if added)
        # plus one more axes for dropout heatmaps
        i = n_expts + extra_cols + 1

        # weights
        exp_weights = cell_weights[cell_weights.experience_level == experience_level]

        # image kernels
        image_weights = []
        for f, feature in enumerate(weights_features[:8]):  # first 8 are images
            image_weights.append(exp_weights[feature + '_weights'].values[0])
        mean_image_weights = np.mean(image_weights, axis=0)

        # GLM output is all resampled to 30Hz now
        frame_rate = 31
        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        ax[i].plot(t_array, mean_image_weights, color=color)
        ax[i].set_ylabel('weight')
        ax[i].set_title('images')
        ax[i].set_xlabel('time (s)')
        ax_to_share = i

        i += 1
        # all other kernels
        for f, feature in enumerate(weights_features[8:]):
            kernel_weights = exp_weights[feature + '_weights'].values[0]
            if feature == 'omissions':
                n_frames_to_clip = int(kernels['omissions']['length'] * frame_rate) + 1
                kernel_weights = kernel_weights[:n_frames_to_clip]
            t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
            ax[i + f].plot(t_array, kernel_weights, color=color)
            ax[i + f].set_ylabel('')
            ax[i + f].set_title(feature)
            ax[i + f].set_xlabel('time (s)')
            ax[i + f].get_shared_y_axes().join(ax[i + f], ax[ax_to_share])

        # except:
        #     print('could not plot GLM kernels for', experience_level)

    # try:
    # plot dropout score heatmaps
    i = n_expts + extra_cols  # change to extra_cols = 4 if running and pupil are added
    # cell_dropouts['cre_line'] = cre_line
    cell_dropouts = cell_dropouts.groupby(['experience_level']).mean()
    if 'ophys_experiment_id' in cell_dropouts.keys():
        cell_dropouts = cell_dropouts.drop(columns='ophys_experiment_id')
    if 'cell_specimen_id' in cell_dropouts.keys():
        cell_dropouts = cell_dropouts.drop(columns='cell_specimen_id')
    cell_dropouts = cell_dropouts[dropout_features]  # order dropouts properly
    dropouts = cell_dropouts.T
    if len(np.where(dropouts < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'
    ax[i] = sns.heatmap(dropouts, cmap=cmap, vmin=vmin, vmax=1, ax=ax[i], cbar=False)
    # ax[i].set_title('coding scores')
    ax[i].set_yticklabels(dropouts.index.values, rotation=0, fontsize=14)
    ax[i].set_xticklabels(dropouts.columns.values, rotation=90, fontsize=14)
    ax[i].set_ylim(0, dropouts.shape[0])
    ax[i].set_xlabel('')

    metadata_string = utils.get_container_metadata_string(dataset.metadata)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.6, wspace=0.7)
    fig.suptitle(str(cell_specimen_id) + '_' + metadata_string, x=0.53, y=1.02,
                 horizontalalignment='center', fontsize=16)

    if save_dir:
        print('saving plot for', cell_specimen_id)
        utils.save_figure(fig, figsize, save_dir, folder, str(cell_specimen_id) + '_' + metadata_string + '_' + data_type)
        print('saved')


####### plot kernels and average responses for individual cell examples #######


def get_stimulus_response_dfs_for_kernel_windows(dataset, kernels, frame_rate):
    '''
    gets stimulus response dataframes for a given dataset using the time windows defined in kernels

    kernels: the 'kernels' data from glm run_params
    frame_rate: frame rate for which GLM results are generated (typically 30Hz)
    '''

    # images
    # get time window used for kernels in model and create sdf for that window
    t_array = get_t_array_for_kernel(kernels, 'image0', frame_rate)
    window = [t_array[0], t_array[-1]]
    idf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                           data_type='filtered_events', event_type='all', load_from_file=True)

    # omissions
    t_array = get_t_array_for_kernel(kernels, 'omissions', frame_rate)
    window = [t_array[0], t_array[-1]]
    odf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                           data_type='filtered_events', event_type='omissions', load_from_file=True)

    # hits and misses
    t_array = get_t_array_for_kernel(kernels, 'hits', frame_rate)
    window = [t_array[0], t_array[-1]]
    cdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                           data_type='filtered_events', event_type='changes', load_from_file=True)

    image_sdf = idf[(idf.is_change == False) & (idf.omitted == False)]
    omission_sdf = odf[ (odf.omitted == True)]
    change_sdf = cdf[(cdf.is_change == True)]

    return image_sdf, omission_sdf, change_sdf


def plot_image_kernels_and_traces_for_cell(cell_specimen_id, dataset,
                                     image_sdf, omission_sdf, change_sdf,
                                     weights_df, kernels, save_dir=None, ax=None):
    """
    plots the average reponse and kernel weights for each image
    Must use `get_stimulus_response_dfs_for_kernel_windows` to get image omission and change stimulus response dfs
    weights_df and kernels can be obtained using `load_GLM_outputs`
    """

    # cre_line = utils.get_abbreviated_cell_type(dataset.metadata['cre_line'])
    cre_line = dataset.metadata['cre_line'].split('-')[0]

    # get weights for example cell
    ophys_experiment_id = dataset.ophys_experiment_id
    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id)
    cell_weights = weights_df[weights_df.identifier == identifier]

    # limit stim response dfs to this cell & relevant conditions (should already be limited to relevant ophys_experiment_id)
    image_cdf = image_sdf[(image_sdf.cell_specimen_id == cell_specimen_id)]
    omission_cdf = omission_sdf[(omission_sdf.cell_specimen_id == cell_specimen_id)]
    hits_cdf = change_sdf[(change_sdf.cell_specimen_id == cell_specimen_id) & (change_sdf.hit == True)]
    misses_cdf = change_sdf[(change_sdf.cell_specimen_id == cell_specimen_id) & (change_sdf.miss == True)]

    # stimulus presentations
    stimulus_presentations = dataset.stimulus_presentations.copy()
    stimulus_presentations = loading.limit_stimulus_presentations_to_change_detection(stimulus_presentations)

    # GLM output is all resampled to 30Hz now
    frame_rate = 31

    # which features to plot
    features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7',
                'omissions', 'hits', 'misses', 'running', 'pupil', 'licks', ]

    color = sns.color_palette()[0]
    if ax is None:
        figsize = (14, 2.5)
        fig, ax = plt.subplots(1, 8, figsize=figsize, sharey=True, )

    for i, feature in enumerate(features[:8]):  # first 8 are images
        kernel_weights = cell_weights[feature + '_weights'].values[0]
        # mean_image_weights = np.mean(image_weights, axis=0)
        # GLM output is all resampled to 30Hz now
        frame_rate = 31
        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=False, alpha=0.2)
        # image_name = image_names[i]
        image_name = stimulus_presentations[stimulus_presentations.image_index == i].image_name.values[0]
        this_image_sdf = image_cdf[image_cdf.image_name == image_name]
        ax[i] = utils.plot_mean_trace(this_image_sdf.trace.values, this_image_sdf.trace_timestamps.values[0],
                                      ylabel='event magnitude', legend_label='data', color=color, interval_sec=0.5,
                                      xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
        ax[i].plot(t_array, kernel_weights, color=color, linestyle='--', label='model')
        ax[i].set_title(feature)
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('')
        # ax[i].set_ylim(-0.01, 0.18)
        # ax[i].get_legend().remove()
        ax_to_share = i
        i += 1
    ax[0].legend(fontsize='x-small')
    ax[0].set_ylabel('Calcium events')
    ymin, ymax = ax[0].get_ylim()
    for x in range(i)[1:]:
        ax[x].set_yticklabels([])

    if save_dir:
        title_string = str(ophys_experiment_id) + '_' + str(cell_specimen_id) + '_' + cre_line
        fig.suptitle(title_string, x=0.5, y=1.15, fontsize=16)
        utils.save_figure(fig, figsize, save_dir, 'example_model_fits', title_string + '_kernels_images')

    return ax


def plot_all_kernels_for_cell(cell_specimen_id, dataset, weights_df, kernels,
                              linewidth=1.5, fontsize=14, label_x=True, save_dir=None, ax=None):
    """
    plots the kernel weights for each kernel type for a given cell, in a row
    weights_df and kernels can be obtained using `load_GLM_outputs` or by loading files directly from the cache
    """

    # cre_line = utils.get_abbreviated_cell_type(dataset.metadata['cre_line'])
    cre_line = dataset.metadata['cre_line'].split('-')[0]

    # get weights for example cell
    ophys_experiment_id = dataset.ophys_experiment_id
    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id)
    cell_weights = weights_df[weights_df.identifier == identifier]


    # # stimulus presentations
    # stimulus_presentations = dataset.stimulus_presentations.copy()
    # stimulus_presentations = loading.limit_stimulus_presentations_to_change_detection(stimulus_presentations)

    # GLM output is all resampled to 30Hz now
    frame_rate = 31

    # which features to plot
    features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'omissions', 'hits', 'misses', 'running', 'pupil', 'licks', ]

    color = sns.color_palette()[0]
    # all other kernels
    if ax is None:
        figsize = (16, 2.5)
        fig, ax = plt.subplots(1, len(features[8:]) + 1, figsize=figsize, sharey=True,
                               gridspec_kw={'width_ratios': [1.5, 3, 2.25, 2.25, 2, 2, 2, ]})  # match axes widths to kernel durations
    print('plotting other kernels')
    # first axis is all-images
    i = 0
    image_weights = []
    for feature in features[:8]:
        kernel_weights = cell_weights[feature + '_weights'].values[0]
        image_weights.append(kernel_weights)
    mean_image_weights = np.mean(image_weights, axis=0)
    # GLM output is all resampled to 30Hz now
    frame_rate = 31
    t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
    ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=False, alpha=0.2)
    ax[i].plot(t_array, mean_image_weights, color=sns.color_palette()[4], linewidth=linewidth)
    ax[i].set_title('images', fontsize=fontsize)
    # ax[i].set_xlabel('Time (s)', fontsize=fontsize)
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')

    # Annotate axes
    xmin, xmax = ax[i].get_xlim()
    ymin, ymax = ax[i].get_ylim()
    # limit y axis to 1/3 of max & label it

    ytop = np.round(ymax * .3, 3)
    ax[i].set_yticks([0, ytop])
    ax[i].set_yticklabels(['', ytop], va='top', fontsize=fontsize-2)
    # # add scale bar for y axis, from 0 to 1/3 of max
    # plot a line for 1/3 of the axis, corresponding to the yticklabel for y axis, which is set at 1/3 of the max y value
    ax[i].axvline(x=t_array[0]-0.25, ymin=0, ymax=0.3, color='k', linewidth=1, clip_on=False)
    ax[i].annotate(str(ytop), xy=(t_array[0] - 0.4, ytop),
                   xycoords='data', xytext=(t_array[0] - 0.4, ytop), ha='right',
                   fontsize=fontsize-2, clip_on=False, annotation_clip=False)

    ax[i].set_xticks([t_array[0], t_array[0] + 0.5])
    # plot time axis bar on x axis
    labelbottom=False
    if label_x:
        ax[i].set_xticklabels(['', '0.5'], va='center', ha='right', fontsize=fontsize-2)
        ax[i].annotate('', xy=(t_array[0], -0.1), xycoords=ax[i].get_xaxis_transform(), xytext=(t_array[0] + 0.5, -0.1),
                       fontsize=fontsize - 2, arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0),
                       clip_on=False, annotation_clip=False)
        labelbottom=True
        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=labelbottom, labeltop=False, labelright=False, labelleft=True)
        ax[i].set_xlabel('images', fontsize=fontsize-1)
        ax[i].set_title('')
    else:
        ax[i].set_xticklabels('')


        # despine
        sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)


    for i, feature in enumerate(features[8:]):
        i += 1
        kernel_weights = cell_weights[feature + '_weights'].values[0]
        if feature == 'omissions':
            n_frames_to_clip = int(kernels['omissions']['length'] * frame_rate) + 1
            kernel_weights = kernel_weights[:n_frames_to_clip]

        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        if feature == 'omissions':
            ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=True, alpha=0.2)
        elif feature == 'hits':
            ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=True, omitted=False, alpha=0.2)
        elif feature == 'misses':
            ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=True, omitted=False, alpha=0.2)
        # plot kernel
        ax[i].plot(t_array, kernel_weights, color=sns.color_palette()[4], linewidth=linewidth)

        ax[i].set_ylabel('')
        ax[i].set_title(feature, fontsize=fontsize)
        ax[i].set_xlabel('')
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')

        ax[i].set_xticks([t_array[0], t_array[0] + 0.5])
        # plot time axis bar on x axis
        labelbottom = False
        if label_x:
            ax[i].set_xticklabels(['', '0.5'], va='center', ha='right', fontsize=fontsize-2)
            ax[i].annotate('', xy=(t_array[0], -0.1), xycoords=ax[i].get_xaxis_transform(),
                           xytext=(t_array[0] + 0.5, -0.1),
                           fontsize=fontsize - 2,
                           arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0),
                           clip_on=False, annotation_clip=False)
            labelbottom = True
            sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
            ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                              labelbottom=labelbottom, labeltop=False, labelright=False, labelleft=False)
            ax[i].set_xlabel(feature, fontsize=fontsize-1)
            ax[i].set_title('')
        else:
            ax[i].set_xticklabels('')
            sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
            ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                              labelbottom=False, labeltop=False, labelright=False, labelleft=False)

        # # add all annotations
    # for i in range(len(features[8:])+1):
        # Annotate axes
        # xmin, xmax = ax[i].get_xlim()
        # ymin, ymax = ax[i].get_ylim()
        # # limit y axis to 1/3 of max & label it
        # ax[i].set_yticks([0, np.round(ymax * .3, 2)])
        # ax[i].set_yticklabels(['', np.round(ymax * .3, 2)], va='top', ha='right', fontsize=fontsize-2)
        #
        # # add scale bar for y axis, from 0 to 1/3 of max
        # ax[i].annotate('', xy=(xmin, 0), xycoords='data', xytext=(xmin, np.round(ymax * 0.3, 2)),
        #                fontsize=fontsize-2, arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), annotation_clip=False)
        # # plot time axis bar on x axis
        # ax[i].set_xticks([t_array[0], t_array[0]+0.5])
        # ax[i].set_xticklabels(['', '0.5 s'], va='top')
        # ax[i].annotate('', xy=(t_array[0], -0.08), xycoords=ax[i].get_xaxis_transform(), xytext=(t_array[0]+0.5, -0.08),
        #                fontsize=fontsize-2, arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), clip_on=False)
        #
        # despine
        # sns.despine(ax=ax[i], top=True, right=True, left=True, bottom=True)
        # ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
        #                   labelbottom=labelbottom, labeltop=False, labelright=False, labelleft=False)

        i += 1

    # ax[i-1].legend(fontsize='x-small')
    ax[0].set_ylabel('Calcium events', fontsize=fontsize)

    if save_dir:
        metadata_string = utils.get_metadata_string(dataset.metadata)
        title_string = metadata_string + '_' + str(cell_specimen_id)
        plt.suptitle(title_string, x=0.5, y=1.15, fontsize=18)
        utils.save_figure(fig, figsize, save_dir, 'example_kernels', title_string + '_kernels')

    return ax


def plot_all_kernels_and_traces_for_cell(cell_specimen_id, dataset, image_sdf, omission_sdf, change_sdf,
                                         weights_df, kernels, show_cell_response=True, save_dir=None, ax=None):
    """
    plots the average reponse and kernel weights for each kernel type for a given cell
    Must use `get_stimulus_response_dfs_for_kernel_windows` to get image omission and change stimulus response dfs
    weights_df and kernels can be obtained using `load_GLM_outputs` or by loading files directly from the cache
    """

    # cre_line = utils.get_abbreviated_cell_type(dataset.metadata['cre_line'])
    cre_line = dataset.metadata['cre_line'].split('-')[0]

    # get weights for example cell
    ophys_experiment_id = dataset.ophys_experiment_id
    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id)
    cell_weights = weights_df[weights_df.identifier == identifier]

    # limit stim response dfs to this cell & relevant conditions (should already be limited to relevant ophys_experiment_id)
    image_cdf = image_sdf[(image_sdf.cell_specimen_id == cell_specimen_id)]
    omission_cdf = omission_sdf[(omission_sdf.cell_specimen_id == cell_specimen_id)]
    hits_cdf = change_sdf[(change_sdf.cell_specimen_id == cell_specimen_id) & (change_sdf.hit == True)]
    misses_cdf = change_sdf[(change_sdf.cell_specimen_id == cell_specimen_id) & (change_sdf.miss == True)]

    # stimulus presentations
    stimulus_presentations = dataset.stimulus_presentations.copy()
    stimulus_presentations = loading.limit_stimulus_presentations_to_change_detection(stimulus_presentations)

    # GLM output is all resampled to 30Hz now
    frame_rate = 31

    # which features to plot
    features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'omissions', 'hits', 'misses', 'running', 'pupil', 'licks', ]

    color = sns.color_palette()[0]
    # all other kernels
    if ax is None:
        figsize = (16, 2.5)
        fig, ax = plt.subplots(1, len(features[8:]) + 1, figsize=figsize, sharey=True,
                               gridspec_kw={'width_ratios': [1, 3, 2.25, 2.25, 2, 2, 2, ]})  # match axes widths to kernel durations
    print('plotting other kernels')
    # first axis is all-images
    i = 0
    image_weights = []
    for feature in features[:8]:
        kernel_weights = cell_weights[feature + '_weights'].values[0]
        image_weights.append(kernel_weights)
    mean_image_weights = np.mean(image_weights, axis=0)
    # GLM output is all resampled to 30Hz now
    frame_rate = 31
    t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
    ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=False, alpha=0.2)
    if show_cell_response:
        ax[i] = utils.plot_mean_trace(image_cdf.trace.values, image_cdf.trace_timestamps.values[0],
                                      ylabel='event magnitude', legend_label='data', color=color, interval_sec=0.5,
                                      xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
    ax[i].plot(t_array, mean_image_weights, color=color, linestyle='--', label='model')
    ax[i].set_title('images')
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('')
    # ax[i].set_ylim(-0.01, 0.18)

    for i, feature in enumerate(features[8:]):
        i += 1
        kernel_weights = cell_weights[feature + '_weights'].values[0]
        if feature == 'omissions':
            n_frames_to_clip = int(kernels['omissions']['length'] * frame_rate) + 1
            kernel_weights = kernel_weights[:n_frames_to_clip]

        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        if feature == 'omissions':
            ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=True, alpha=0.2)
            if show_cell_response:
                ax[i] = utils.plot_mean_trace(omission_cdf.trace.values, omission_cdf.trace_timestamps.values[0],
                                          ylabel='event magnitude', legend_label='data', color=color, interval_sec=1,
                                          xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
        elif feature == 'hits':
            try:
                ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=True, omitted=False, alpha=0.2)
                if show_cell_response:
                    ax[i] = utils.plot_mean_trace(hits_cdf.trace.values, hits_cdf.trace_timestamps.values[0],
                                              ylabel='event magnitude', legend_label='data', color=color, interval_sec=1,
                                              xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
            except:
                print('could not plot traces for', feature, ophys_experiment_id, cell_specimen_id)
        elif feature == 'misses':
            try:
                ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=True, omitted=False, alpha=0.2)
                if show_cell_response:
                    ax[i] = utils.plot_mean_trace(misses_cdf.trace.values, misses_cdf.trace_timestamps.values[0],
                                              ylabel='event magnitude', legend_label='data', color=color, interval_sec=1,
                                              xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
            except:
                print('could not plot traces for', feature, ophys_experiment_id, cell_specimen_id)
        try:
            ax[i].plot(t_array, kernel_weights, color=color, linestyle='--', label='model')

        except:
            print('could not plot kernel weights for', feature, ophys_experiment_id, cell_specimen_id)
        ax[i].set_ylabel('')
        ax[i].set_title(feature)
        ax[i].set_xlabel('Time (s)')
        # ax[i].set_ylim(-0.01, 0.18)
        # match ylims to image kernels
        # ax[i].set_ylim(ymin, ymax)
        # ax[i].get_shared_y_axes().join(ax[i + f], ax[ax_to_share])
        i += 1
    ax[i-1].legend(fontsize='x-small')
    ax[0].set_ylabel('Calcium events')

    if save_dir:
        title_string = str(ophys_experiment_id) + '_' + str(cell_specimen_id) + '_' + cre_line
        plt.suptitle(title_string, x=0.5, y=1.15, fontsize=18)
        utils.save_figure(fig, figsize, save_dir, 'example_model_fits', title_string + '_kernels_other')

    return ax


def plot_kernels_traces_and_dropouts_examples_for_experiment(ophys_experiment_id, run_params, all_results,
                                                             results_pivoted, weights_df,
                                                             use_var_exp=True, save_dir=None):
    '''
    Plot kernels and cell traces overlaid for high variance explained cells
    Also plot (on separate axes) coding scores for the same cells
    '''

    if use_var_exp:  # get cells with high variance explained
        print('filtering cells by variance explained')
        expt_data = all_results[(all_results.variance_explained_full > 0.25) & (all_results.ophys_experiment_id == ophys_experiment_id)]
        cell_specimen_ids = expt_data.cell_specimen_id.unique()
        traces = []
    else:  # get high SNR traces based on dff signal (this takes longer because you have to load dataset)
        print('getting high SNR cells')
        # load dataset & filter stim presentations
        dataset = loading.get_ophys_dataset(ophys_experiment_id)
        # get high SNR traces
        traces = processing.compute_robust_snr_on_dataframe(dataset.dff_traces)

        if len(traces) < 10:
            n_cells = len(traces)
        else:
            n_cells = 10
        cell_specimen_ids = traces.sort_values(by='robust_snr', ascending=False).index.values[:n_cells]
        expt_data = all_results[(all_results.ophys_experiment_id == ophys_experiment_id)]

    # generate the plots
    cre_line = expt_data.cre_line.unique()[0].split('-')[0]
    print(len(cell_specimen_ids), 'cells for ophys_experiment_id: ', ophys_experiment_id, ', cre:', cre_line)
    if len(cell_specimen_ids) > 3:
        if len(traces) == 0:  # o0nly get dataset if it hasnt already been loaded to get high SNR cells
            # load dataset & filter stim presentations
            dataset = loading.get_ophys_dataset(ophys_experiment_id)

        # get stimulus response dfs for kernel lengths
        kernels = run_params['kernels']
        frame_rate = 31
        image_sdf, omission_sdf, change_sdf = get_stimulus_response_dfs_for_kernel_windows(dataset, kernels, frame_rate)

        # loop through cells and plot
        for cell_specimen_id in cell_specimen_ids:
            print('cell_specimen_id =', cell_specimen_id)
            plot_image_kernels_and_traces_for_cell(cell_specimen_id, dataset, image_sdf, omission_sdf, change_sdf, weights_df, kernels, save_dir=save_dir)
            plot_all_kernels_and_traces_for_cell(cell_specimen_id, dataset, image_sdf, omission_sdf, change_sdf, weights_df, kernels, save_dir=save_dir)

            plot_coding_scores_for_cell(cell_specimen_id, ophys_experiment_id, results_pivoted, save_dir=save_dir)


######## coding scores plots ###########

def plot_coding_scores_for_cell(cell_specimen_id, ophys_experiment_id, results_pivoted, save_dir=None, ax=None):
    '''
    Creates barplot of coding scores for a single cell in a single experiment and saves it
    '''

    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id)
    # get dropouts just for one cell
    cell_dropouts = results_pivoted[results_pivoted.identifier == identifier]

    # which features to plot
    coding_score_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'all-images',
                             'omissions', 'hits', 'misses', 'task', 'running', 'pupil', 'licks', 'behavioral']

    if ax is None:
        figsize = (3, 6)
        fig, ax = plt.subplots(figsize=figsize)
    # get dropouts just for one cell
    ax = sns.barplot(data=np.abs(cell_dropouts[coding_score_features]), orient='h', color='gray', ax=ax)  # color=sns.color_palette('Blues_r')[0], ax=ax)
    ax.set_xlabel('Coding score')
    ax.set_title('var_exp_full = ' + str(np.round(cell_dropouts.variance_explained_full.values[0], 3)))

    if save_dir:
        # cre_line = utils.get_abbreviated_cell_type(cell_dropouts.cre_line.values[0])
        cre_line = cell_dropouts['cre_line'].values[0].split('-')[0]
        title_string = str(ophys_experiment_id) + '_' + str(cell_specimen_id) + '_' + cre_line
        fig.suptitle(title_string, x=0.5, y=1., fontsize=16)
        utils.save_figure(fig, figsize, save_dir, 'example_model_fits', title_string + '_coding_scores')

    return ax


def plot_coding_score_components_for_cell(cell_specimen_id, ophys_experiment_id, results_pivoted,
                                          fontsize=12, as_panel=False, save_dir=None, ax=None):
    '''
    Creates barplot of coding scores for a single cell in a single experiment and saves it
    '''
    import visual_behavior.dimensionality_reduction.clustering.plotting as plotting

    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id)
    # get dropouts just for one cell
    cell_dropouts = results_pivoted[results_pivoted.identifier == identifier]

    # which features to plot
    main_coding_score_features = ['all-images', 'omissions', 'task', 'behavioral']
    feature_labels = ['images', 'omissions', 'task', 'behavior']
    feature_colors, feature_labels_dict = plotting.get_feature_colors_and_labels()

    if ax is None:
        figsize = (3, 2)
        fig, ax = plt.subplots(figsize=figsize)
    # get dropouts just for one cell
    ax = sns.barplot(data=np.abs(cell_dropouts[main_coding_score_features]), orient='v',
                     palette=feature_colors, ax=ax)  # color=sns.color_palette('Blues_r')[0], ax=ax)
    ax.set_ylabel('Coding score', fontsize=fontsize)
    ax.set_title('var_exp_full = ' + str(np.round(cell_dropouts.variance_explained_full.values[0], 3)), fontsize=fontsize)
    ax.set_xticklabels(feature_labels, rotation=45, horizontalalignment='right', fontsize=fontsize+1)

    for x, feature in enumerate(main_coding_score_features):
        cs = np.abs(cell_dropouts[feature].values[0])
        if cs>0:
            ax.text(s = str(np.round(cs,2)), x=x, y=cs, rotation=0, fontsize=fontsize-1, color='k', va='bottom',  ha='center')
    if as_panel:
        ax.set_title('')
        # Annotate axes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # limit y axis to 1/3 of max & label it
        # ytop = np.round(ymax * .3, 2)
        # ytop = 1
        # ax.set_yticks([0, ytop])
        ax.set_ylim(0, 1)
        ax.set_yticklabels('')
        # ax.set_yticklabels([0, ytop], va='top', fontsize=fontsize)
        #
        # ax.annotate('', xy=(xmin-0.1, 0), xycoords='data', xytext=(xmin-0.1, ytop),
        #                fontsize=fontsize, arrowprops=dict(arrowstyle='-', color='k', lw=1, shrinkA=0, shrinkB=0), annotation_clip=False)

        # plot time axis bar on x axis
        # despine
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
        ax.tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=True, labeltop=False, labelright=False, labelleft=True)


    if save_dir:
        # cre_line = utils.get_abbreviated_cell_type(cell_dropouts.cre_line.values[0])
        cre_line = cell_dropouts['cre_line'].values[0].split('-')[0]
        title_string = str(ophys_experiment_id) + '_' + str(cell_specimen_id) + '_' + cre_line
        fig.suptitle(title_string, x=0.5, y=1., fontsize=16)
        utils.save_figure(fig, figsize, save_dir, 'example_model_fits', title_string + '_coding_scores')

    return ax


######## plot model fits, kernels, and coding scores in the same figure #######


def plot_glm_example_cells_figure(ophys_experiment_id, run_params,
                                  all_results, results_pivoted, weights_df,
                                  cell_specimen_ids=None, use_var_exp=True, save_dir=None):


    # load dataset & filter stim presentations
    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    # get GLM results
    cell_results_df, expt_results, dropouts = load_glm_model_fit_results(ophys_experiment_id)

    if cell_specimen_ids is None: # if no cell specimen ids are provided,
        # get cells to plot, either based on variance explained or trace SNR
        if use_var_exp:  # get cells with high variance explained
            filter_type = '_high_variance_explained'
            print('filtering cells by variance explained')
            expt_data = all_results[(all_results.variance_explained_full > 0.1) & (all_results.ophys_experiment_id == ophys_experiment_id)]
            cell_specimen_ids = expt_data.cell_specimen_id.unique()
            print('there are', len(cell_specimen_ids), 'cells with variance_explained_full > 0.1 in experiment', ophys_experiment_id)
            traces = []
        else:  # get high SNR traces based on dff signal (this takes longer because you have to load dataset)
            filter_type = '_high_trace_snr'
            print('getting high SNR cells')
            # load dataset & filter stim presentations
            dataset = loading.get_ophys_dataset(ophys_experiment_id)
            # get high SNR traces
            traces = processing.compute_robust_snr_on_dataframe(dataset.dff_traces)
            cell_specimen_ids = traces.sort_values(by='robust_snr', ascending=False).index.values
            expt_data = all_results[(all_results.ophys_experiment_id == ophys_experiment_id)]
            print('there are', len(cell_specimen_ids), 'high SNR cells in experiment', ophys_experiment_id)
        # limit to 10 cells per exp
        if len(cell_specimen_ids) < 10:
            n_cells = len(cell_specimen_ids)
        else:
            n_cells = 10
        cell_specimen_ids = cell_specimen_ids[:n_cells]

    if len(cell_specimen_ids) > 0:

        print('loading stimulus response dfs')
        # get stimulus response dfs for kernel lengths
        kernels = run_params['kernels']
        frame_rate = 31
        image_sdf, omission_sdf, change_sdf = get_stimulus_response_dfs_for_kernel_windows(dataset, kernels, frame_rate)

        print('generating figure')
        # make multi panel figure
        for cell_specimen_id in cell_specimen_ids:

            figsize = [2 * 11, 2 * 8.5]
            fig = plt.figure(figsize=figsize, facecolor='white')

            ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.45), yspan=(0, 0.15))
            ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                            None, include_events=True, save_dir=None, ax=ax)
            ax.set_xlabel('')

            ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.45), yspan=(0.2, 0.35))
            ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                            kernel='all-images', include_events=True, save_dir=None, ax=ax)
            ax.set_title('')
            ax.set_xlabel('')

            ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.45), yspan=(0.4, 0.55))
            ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                            kernel='omissions', include_events=True, save_dir=None, ax=ax)
            ax.set_title('')

            ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.55, 1.0), yspan=(0, 0.15))
            ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                            kernel=None, include_events=False, save_dir=None, ax=ax)
            ax.set_xlabel('')

            ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.55, 1.0), yspan=(0.2, 0.35))
            ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                            kernel='all-images', include_events=False, save_dir=None, ax=ax)
            ax.set_title('')
            ax.set_xlabel('')

            ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.55, 1.0), yspan=(0.4, 0.55))
            ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results,
                                            kernel='omissions', include_events=False, save_dir=None, ax=ax)
            ax.set_title('')


            ax = utils.placeAxesOnGrid(fig, dim=(1, 8), xspan=(0., 0.7), yspan=(0.65, 0.75), sharey=True)
            ax = plot_image_kernels_and_traces_for_cell(cell_specimen_id, dataset, image_sdf, omission_sdf, change_sdf,
                                                            weights_df, kernels, save_dir=None, ax=ax)

            ax = utils.placeAxesOnGrid(fig, dim=(1, 7), xspan=(0., 0.7), yspan=(0.9, 1.0), sharey=True,)
                                    #    gridspec_kw={'width_ratios': [1, 3, 2.25, 2.25, 2, 2, 2, ]})
            ax = plot_all_kernels_and_traces_for_cell(cell_specimen_id, dataset, image_sdf, omission_sdf, change_sdf,
                                                    weights_df, kernels, save_dir=None, ax=ax)

            ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.8, 1.0), yspan=(0.65, 1.0))
            ax = plot_coding_scores_for_cell(cell_specimen_id, ophys_experiment_id, results_pivoted, save_dir=None, ax=ax)

            if save_dir:
                metadata_string = utils.get_metadata_string(dataset.metadata)
                fig.suptitle(metadata_string, x=0.5, y=0.9, fontsize = 16)
                cre_line = dataset.metadata['cre_line'].split('-')[0]
                utils.save_figure(fig, figsize, save_dir, 'glm_model_fits_figures'+filter_type,
                            str(ophys_experiment_id) + '_' + str(cell_specimen_id) + '_' + cre_line)


def plot_model_fits_and_kernels_for_example_cell(ophys_experiment_id, cell_specimen_id, dataset,
                                                     image_sdf, omission_sdf, change_sdf,
                                                     results_pivoted, weights_df, kernels, save_dir=None):
    '''
    Create multi panel figure with:
        cell events trace + model fit
        cell events trace + model fit with images removed
        cell events trace + model fit with omissions removed
        cell dff trace + model fit
        cell dff trace + model fit with images removed
        cell dff trace + model fit with omissions removed
        kernels for this cell
        coding scores for this cell
    '''

    # get GLM results
    cell_results_df, expt_results, dropouts = load_glm_model_fit_results(ophys_experiment_id)

    # plot figuregep.
    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.45), yspan=(0, 0.15))
    ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results, None, include_events=True, save_dir=None, ax=ax)
    ax.set_xlabel('')

    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.45), yspan=(0.2, 0.35))
    ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results, kernel='all-images', include_events=True, save_dir=None, ax=ax)
    ax.set_title('')
    ax.set_xlabel('')

    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.45), yspan=(0.4, 0.55))
    ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results, kernel='omissions', include_events=True, save_dir=None, ax=ax)
    ax.set_title('')

    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.55, 1.0), yspan=(0, 0.15))
    ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results, kernel=None, include_events=False, save_dir=None, ax=ax)
    ax.set_xlabel('')

    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.55, 1.0), yspan=(0.2, 0.35))
    ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results, kernel='all-images', include_events=False, save_dir=None, ax=ax)
    ax.set_title('')
    ax.set_xlabel('')

    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.55, 1.0), yspan=(0.4, 0.55))
    ax = plot_model_fits_example_cell(cell_specimen_id, dataset, cell_results_df, dropouts, expt_results, kernel='omissions', include_events=False, save_dir=None, ax=ax)
    ax.set_title('')

    ax = utils.placeAxesOnGrid(fig, dim=(1, 8), xspan=(0., 0.7), yspan=(0.65, 0.75), sharey=True)
    ax = plot_image_kernels_and_traces_for_cell(cell_specimen_id, dataset, image_sdf, omission_sdf, change_sdf, weights_df, kernels, save_dir=None, ax=ax)

    ax = utils.placeAxesOnGrid(fig, dim=(1, 7), xspan=(0., 0.7), yspan=(0.9, 1.0), sharey=True, )
    ax = plot_all_kernels_and_traces_for_cell(cell_specimen_id, dataset,
                                              image_sdf, omission_sdf, change_sdf,
                                              weights_df, kernels, show_cell_response=False, save_dir=None, ax=ax)

    ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.8, 1.0), yspan=(0.65, 1.0))
    ax = plot_coding_scores_for_cell(cell_specimen_id, ophys_experiment_id, results_pivoted, save_dir=None, ax=ax)

    metadata_string = utils.get_metadata_string(dataset.metadata)
    fig.suptitle(metadata_string, x=0.5, y=0.9, fontsize=16)

    if save_dir:
        folder = 'glm_model_fits_figures'
        metadata_string = utils.get_metadata_string(dataset.metadata)
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(cell_specimen_id))



##### plot weights of all cells plus population averages #######

def plot_weights_and_coding_score_heatmaps_for_experience_levels(kernel, weights_df, run_params, row_condition='cre_line', col_condition='experience_level', vmax=0.002, xlabel='Time (s)',
                                                                 save_dir=None, folder=None):
    '''
    Plot a heatmap of kernel weights and coding scores for all cells
    Default values results in a figure with experience levels as columns and cre lines as rows

    kernel: str, name of kernel to plot, must be column in weights_df
    weights_df: dataframe of kernel weights for all cells & cell types
    run_params: dict of run param values, including kernel durations
    row_condition: column in weights_df to iterate over for rows
    col_condition: column in weights_df to iterate over for columns
    vmax: value of max and min weights to plot
    xlabel: string to label xaxis with
    '''

    # get weights for this kernel
    kernel_weights = weights_df[[kernel + '_weights', kernel, row_condition, col_condition]]

    # conditions to iterate over
    row_conditions = np.sort(kernel_weights[row_condition].unique())
    col_conditions = np.sort(kernel_weights[col_condition].unique())

    # get timestamps and intervals for xticklabels
    timestamps = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'], 1 / 31.)
    timestamps = np.round(timestamps, 2)
    if 'image' in kernel:
        timestamps = timestamps[:-1]
    if np.max(timestamps) <= 1:
        interval_sec = 0.5
    else:
        interval_sec = 1
    xlim_seconds = [timestamps[0], timestamps[-1]]

    # settings for population average plot
    if kernel == 'omissions':
        omitted = True
    else:
        omitted = False
    if (kernel == 'hits') or (kernel == 'misses'):
        change = True
    else:
        change = False
    palette = utils.get_experience_level_colors()

    # coding score colormap
    try: # newer version of mpl
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap('Blues')
    except: # older version
        cmap = plt.get_cmap('Blues')
    cmap.set_under('black')

    # create plot
    figsize = [16, 5]
    fig = plt.figure(figsize=figsize, facecolor='white')
    # loop through experience levels
    for c, col in enumerate(col_conditions):
        col_weights = kernel_weights[kernel_weights[col_condition] == col]

        # create axes for this experience level
        xspan = ((c * .1) * 2.5, ((c * .1) * 2.5) + 0.19)
        print(xspan)
        ax = utils.placeAxesOnGrid(fig, dim=(len(row_conditions), 2), xspan=xspan, yspan=(0, 1), wspace=0, hspace=0, sharex='col', sharey='row', width_ratios=[8, 1])

        # loop through cre lines
        for r, row in enumerate(row_conditions):
            row_weights = col_weights[col_weights[row_condition] == row]
            row_weights = row_weights.sort_values(by=kernel)
            row_weights = row_weights.dropna()

            # weights
            data = pd.DataFrame(np.vstack(row_weights[kernel + '_weights'].values), columns=timestamps)
            cbar_weights = ax[r][0].imshow(data.values, aspect='auto', cmap='PRGn', vmin=-vmax, vmax=vmax, extent=[timestamps[0], timestamps[-1], 0, np.shape(data)[0]])
            ax[r][0].set_yticks([0, len(data)])
            ax[r][0].set_yticklabels([len(data), ''])
            if c == 0:
                ax[r][0].set_ylabel(utils.convert_cre_line_to_cell_type(row))
                ax[r][0].set_ylabel(utils.convert_cre_line_to_cell_type(row).split(' ')[0])
            else:
                ax[r][0].set_ylabel('')
            if r < 2:
                ax[r][0].set_xticklabels([])
            else:
                ax[r][0].set_xticks(np.arange(timestamps[0], timestamps[-1], interval_sec))
                ax[r][0].set_xlabel(xlabel)

            # coding scores
            row_weights = col_weights[col_weights[row_condition] == row]
            row_weights = row_weights.sort_values(by=kernel, ascending=True)
            # data = pd.DataFrame(np.sqrt(row_weights[kernel].values))
            coding_scores = row_weights[kernel].values
            coding_scores = np.sqrt(coding_scores[:, np.newaxis] * -1)
            cbar_coding = ax[r][1].imshow(coding_scores, aspect='auto', cmap=cmap, vmin=1e-10, vmax=1)
            ax[r][1].set_yticks([0, len(data)])
            ax[r][1].set_xticks([])
            ax[r][1].set_yticklabels([])
            # ax[r][1].set_ylabel(utils.convert_cre_line_to_cell_type(row))
            ax[r][1].set_xticklabels([])

        ax[0][0].set_title(utils.convert_experience_level(col), color=palette[c])

    last_x_val = ((c * .1) * 2.5) + 0.2
    xspan = (last_x_val, last_x_val + 0.02)
    # weights colorbar
    cax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=xspan, yspan=(0.4, 0.6))
    color_bar = fig.colorbar(cbar_weights, cax=cax)
    color_bar.ax.set_title('Weight', fontsize=16, loc='left')
    # color_bar.set_clim(zlims[0], zlims[1])
    color_bar.ax.tick_params(axis='both', labelsize=14)
    # coding scores colorbar
    cax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=xspan, yspan=(0.8, 1))
    color_bar = fig.colorbar(cbar_coding, cax=cax, extend='min')
    color_bar.ax.set_title('Coding \nscore', fontsize=16, loc='left')
    color_bar.set_ticks([0, .5, 1])
    color_bar.ax.tick_params(axis='both', labelsize=14)

    # create axes for population averages
    c += 1
    xspan = (((c * .1) * 2.5) + 0.04, ((c * .1) * 2.5) + 0.23)
    print(xspan)
    ax = utils.placeAxesOnGrid(fig, dim=(len(row_conditions), 1), xspan=xspan, yspan=(0, 1), wspace=0, hspace=0, sharex='col')

    # plot population average kernels for each cre line & experience level
    for r, row in enumerate(row_conditions):
        for c, col in enumerate(col_conditions):
            # get weights for this cre line & experience level
            condition_weights = kernel_weights[(kernel_weights[row_condition] == row) & (kernel_weights[col_condition] == col)]
            condition_weights = condition_weights.dropna()
            traces = np.vstack(np.asarray(condition_weights[kernel + '_weights']))
            # plot average trace
            ax[r] = utils.plot_mean_trace(traces, timestamps, ylabel='Kernel weights', legend_label=col, color=palette[c], interval_sec=interval_sec, xlim_seconds=xlim_seconds, ax=ax[r])
            ax[r] = utils.plot_flashes_on_trace(ax[r], timestamps, change=change, omitted=omitted)
            # ax.legend(loc='upper right', fontsize='xx-small')
        ax[r].set_xlim(xlim_seconds)
        ax[r].yaxis.set_label_position("right")
        ax[r].yaxis.tick_right()
        if r < 2:
            ax[r].set_xlabel('')
            ax[r].set_xticks([])
        if r != 1:
            ax[r].set_ylabel('')
        sns.despine(ax=ax[r], left=True, bottom=False, right=False, )
    ax[0].set_title('Average weights')
    ax[r].set_xlabel(xlabel)

    plt.suptitle(kernel + ' kernels', x=0.5, y=1)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, 'weight_heatmaps', kernel)


if __name__ == '__main__':

    import visual_behavior_glm.GLM_fit_dev as gfd
    import visual_behavior_glm.GLM_visualization_tools as gvt
    import visual_behavior_glm.GLM_analysis_tools as gat

    import visual_behavior.visualization.ophys.platform_paper_figures as ppf

    # load metadata tables for platform experiments
    platform_experiments = loading.get_platform_paper_experiment_table(limit_to_closest_active=True)
    cells_table = loading.get_cell_table(platform_paper_only=True)

    print(len(platform_experiments))  # should be 402

    # save directory
    save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_plots\figure_3'

    # load GLM results for platform paper experiments

    # Define model version
    glm_version = '24_events_all_L2_optimize_by_session'

    # Load GLM data
    run_params, results, results_pivoted, weights_df = gfd.load_analysis_dfs(glm_version)

    # limit to platform expts
    results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(platform_experiments.index.values)]

    # weights limited to platform expts
    weights_df = weights_df[weights_df.ophys_experiment_id.isin(platform_experiments.index.values)]

    # Plot kernel activations for specfic experiment

    ophys_experiment_id = 808621034
    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    start_time = 2318  # 1500, 1450, 2550, 2320
    duration_seconds = 19.5
    plot_behavior_timeseries_and_GLM_kernel_activations(dataset, start_time, duration_seconds, save_dir=save_dir)

    # Plot kernels and coding scores for a specific cell
    cell_specimen_id = 1086501664
    ophys_experiment_id = 849233396
    identifier = str(ophys_experiment_id) + '_' + str(cell_specimen_id)

    # get weights for example cell
    cell_weights = weights_df[weights_df.identifier == identifier]
    # get dropouts just for one cell
    cell_dropouts = results_pivoted[results_pivoted.identifier == identifier]
    # get all results for this cell
    cell_results = results[results.identifier == identifier]

    kernels = run_params['kernels']
    frame_rate = 31

    # get dataset and average responses
    dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)

    image_sdf, omission_sdf, change_sdf = get_stimulus_response_dfs_for_kernel_windows(dataset, kernels, frame_rate)

    # plot kernels and coding scores
    plot_kernels_and_traces_for_cell(cell_specimen_id, dataset,
                                     image_sdf, omission_sdf, change_sdf,
                                     cell_weights, run_params, save_dir=save_dir)

    plot_coding_scores_for_cell(cell_specimen_id, ophys_experiment_id, results_pivoted, save_dir=save_dir)
