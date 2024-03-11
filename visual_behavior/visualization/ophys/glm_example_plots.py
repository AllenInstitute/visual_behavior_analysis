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
import visual_behavior.visualization.ophys.summary_figures as sf

import mindscope_utilities.general_utilities as ms_utils

import visual_behavior_glm.GLM_params as glm_params

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
# sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
sns.set_palette('deep')


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
    # get all images & assign colors (image colors wont be used if a color is provided or if label_changes is True)
    images = np.sort(stim_table[stim_table.omitted == False].image_name.unique())
    image_colors = sns.color_palette("hls", len(images))
    # limit to time window
    stim_table = stim_table[(stim_table.start_time >= xlim_seconds[0]) & (stim_table.stop_time <= xlim_seconds[1])]
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
            is_hit = trials[trials.change_frame == stim_table.loc[idx].start_frame - 1].hit.values[0]
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
    ax.set_ylabel(kernel, rotation=0, horizontalalignment='right', verticalalignment='center')

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

    n_rows = 10
    figsize = (10, 6)
    fig, ax = plt.subplots(n_rows, 1, figsize=figsize, sharex=True,
                           gridspec_kw={'height_ratios': [1.5, 1, 1, 1, 1, 1, 1, 1, 1.5, 1.5, ]})
    ax = ax.ravel()

    colors = sns.color_palette()

    # first axis is licking behavior + stim timing
    ax[0] = ppf.add_stim_color_span(dataset, ax[0], xlim=xlim_seconds)
    ax[0].plot(lick_timestamps, licks, '|', label='licks', color='gray', markersize=10)
    ax[0].plot(reward_timestamps, rewards, '^', label='rewards', color='b', markersize=8)
    ax[0].set_yticklabels([])
    ax[0].set_ylim(-3, 0)
    # ax[0].legend(bbox_to_anchor=(0, 0.529))
    ax[0].legend(bbox_to_anchor=(0, 0.529))
    # ax[0].set_ylabel('stimulus & behavior events')

    # next axes are the kernel timings
    ax[1] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='image 1', ax=ax[1])
    ax[2] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='image 2', ax=ax[2])
    ax[3] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='image 3', ax=ax[3])
    ax[4] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='omissions', ax=ax[4])
    ax[5] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='hits', ax=ax[5])
    ax[6] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='misses', ax=ax[6])
    ax[7] = plot_kernel_activations(dataset, start_time, duration_seconds, kernel='licks', ax=ax[7])
    for i in range(1, 8):
        ax[i].set_ylim(-0.2, 1.2)

    # now plot running and pupil as continuous variables
    ax[8].plot(running_timestamps, running_speed, label='running_speed', color='k')
    ax[8].set_ylabel('running', rotation=0, horizontalalignment='right', verticalalignment='center')
    ax[8].set_ylim(ymin=-8)

    ax[9].plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color='k')
    ax[9].set_ylabel('pupil', rotation=0, horizontalalignment='right', verticalalignment='center')
    ax[9].set_xticks(np.arange(xlim_seconds[0], xlim_seconds[1], 4))

    # remove ticks everywhere but bottom row + set xlim for all axes
    i = n_rows - 1
    ax[i].set_xlabel('Time in session (seconds)')
    for i in range(n_rows):
        ax[i].set_xlim(xlim_seconds)
        ax[i].tick_params(which='both', bottom=False, top=False, right=False, left=False,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)
        ax[i].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax[i].set_yticklabels([])
        # ax[i].yaxis.set_label_position("right")()
        if label_stim_times_on_all_rows:
            ax[i] = ppf.add_stim_color_span(dataset, ax[i], xlim=xlim_seconds)
    # bottom row has ticks for timestamps
    ax[n_rows - 1].tick_params(which='both', bottom=True, top=False, right=False, left=False,
                               labelbottom=True, labeltop=False, labelright=False, labelleft=True)

    # add title to top row
    metadata_string = utils.get_metadata_string(dataset.metadata)
    plt.suptitle(metadata_string + '\nEncoding model features', x=0.5, y=1.05, fontsize=18)
    # ax[0].set_title('Encoding model features')

    plt.subplots_adjust(hspace=0, wspace=0.9)

    if save_dir:
        folder = 'behavior_physio_timeseries_GLM_kernels'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)),
                          formats=['.png'])
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

    image_sdf = idf[(idf.is_change == False) & (idf.omitted==False)]
    omission_sdf = odf[ (odf.omitted==True)]
    change_sdf = cdf[(cdf.is_change==True)]

    return image_sdf, omission_sdf, change_sdf


def plot_kernels_and_traces_for_cell(cell_specimen_id, dataset,
                                     image_sdf, omission_sdf, change_sdf,
                                     cell_weights, run_params, save_dir=None):
    """
    plots the average reponse and kernel weights for each kernel type for a given cell
    """

    cre_line = utils.get_abbreviated_cell_type(dataset.metadata['cre_line'])

    # get weights for example cell
    identifier = str(dataset.ophys_experiment_id) + '_' + str(cell_specimen_id)
    cell_weights = weights_df[weights_df.identifier == identifier]

    # limit stim response dfs to this cell & relevant conditions
    image_cdf = image_sdf[(image_sdf.cell_specimen_id == cell_specimen_id)]
    omission_cdf = omission_sdf[(omission_sdf.cell_specimen_id == cell_specimen_id)]
    hits_cdf = change_sdf[(change_sdf.cell_specimen_id == cell_specimen_id) & (change_sdf.hit == True)]
    misses_cdf = change_sdf[(change_sdf.cell_specimen_id == cell_specimen_id) & (change_sdf.miss == True)]

    exp_weights = cell_weights.copy()
    kernels = run_params['kernels']

    # GLM output is all resampled to 30Hz now
    frame_rate = 31

    # which features to plot
    features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7',
                'omissions', 'hits', 'misses', 'running', 'pupil', 'licks', ]

    color = sns.color_palette()[0]
    figsize = (15, 2.5)
    fig, ax = plt.subplots(1, 8, figsize=figsize, sharey=True, )
    for i, feature in enumerate(features[:8]):  # first 8 are images
        kernel_weights = exp_weights[feature + '_weights'].values[0]
        # mean_image_weights = np.mean(image_weights, axis=0)
        # GLM output is all resampled to 30Hz now
        frame_rate = 31
        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=False, alpha=0.2)
        # image_name = image_names[i]
        st = dataset.stimulus_presentations.copy()
        image_name = st[st.image_index == i].image_name.values[0]
        this_image_sdf = image_cdf[image_cdf.image_name == image_name]
        ax[i] = utils.plot_mean_trace(this_image_sdf.trace.values, this_image_sdf.trace_timestamps.values[0],
                                      ylabel='event magnitude', legend_label='data', color=color, interval_sec=0.5,
                                      xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
        ax[i].plot(t_array, kernel_weights, color=color, linestyle='--', label='model')
        ax[i].set_title(feature)
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('')
        ax[i].set_ylim(-0.01, 0.18)
        # ax[i].get_legend().remove()
        ax_to_share = i
        i += 1
    ax[0].legend(fontsize='x-small')
    ax[0].set_ylabel('Calcium events')
    ymin, ymax = ax[0].get_ylim()

    if save_dir:
        title_string = 'oeid_' + str(ophys_experiment_id) + '_csid_' + str(cell_specimen_id) + '_' + cre_line
        fig.suptitle(title_string, x=0.5, y=1.15)
        utils.save_figure(fig, figsize, save_dir, 'kernel_example_plots', title_string + '_image_kernels')

    # all other kernels
    figsize = (16, 2.5)
    fig, ax = plt.subplots(1, len(features[8:]) + 1, figsize=figsize, sharey=True,
                           gridspec_kw={
                               'width_ratios': [1, 3, 2.25, 2.25, 2, 2, 2, ]})  # match axes widths to kernel durations
    # first axis is all-images
    i = 0
    image_weights = []
    for feature in features[:8]:
        kernel_weights = exp_weights[feature + '_weights'].values[0]
        image_weights.append(kernel_weights)
    mean_image_weights = np.mean(image_weights, axis=0)
    # GLM output is all resampled to 30Hz now
    frame_rate = 31
    t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
    ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=False, alpha=0.2)
    # image_name = image_names[i]
    st = dataset.stimulus_presentations.copy()
    ax[i] = utils.plot_mean_trace(image_cdf.trace.values, image_cdf.trace_timestamps.values[0],
                                  ylabel='event magnitude', legend_label='data', color=color, interval_sec=0.5,
                                  xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
    ax[i].plot(t_array, mean_image_weights, color=color, linestyle='--', label='model')
    ax[i].set_title('images')
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('')
    ax[i].set_ylim(-0.01, 0.18)

    for i, feature in enumerate(features[8:]):
        i += 1
        kernel_weights = exp_weights[feature + '_weights'].values[0]
        if feature == 'omissions':
            n_frames_to_clip = int(kernels['omissions']['length'] * frame_rate) + 1
            kernel_weights = kernel_weights[:n_frames_to_clip]

        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        if feature == 'omissions':
            ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=False, omitted=True, alpha=0.2)
            ax[i] = utils.plot_mean_trace(omission_cdf.trace.values, omission_cdf.trace_timestamps.values[0],
                                          ylabel='event magnitude', legend_label='data', color=color, interval_sec=1,
                                          xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
        elif feature == 'hits':
            ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=True, omitted=False, alpha=0.2)
            ax[i] = utils.plot_mean_trace(hits_cdf.trace.values, hits_cdf.trace_timestamps.values[0],
                                          ylabel='event magnitude', legend_label='data', color=color, interval_sec=1,
                                          xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
        elif feature == 'misses':
            ax[i] = utils.plot_flashes_on_trace(ax[i], t_array, change=True, omitted=False, alpha=0.2)
            ax[i] = utils.plot_mean_trace(misses_cdf.trace.values, misses_cdf.trace_timestamps.values[0],
                                          ylabel='event magnitude', legend_label='data', color=color, interval_sec=1,
                                          xlim_seconds=[t_array[0], t_array[-1]], plot_sem=True, ax=ax[i])
        ax[i].plot(t_array, kernel_weights, color=color, linestyle='--', label='model')
        ax[i].set_ylabel('')
        ax[i].set_title(feature)
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylim(-0.01, 0.18)
        # match ylims to image kernels
        # ax[i].set_ylim(ymin, ymax)
        # ax[i].get_shared_y_axes().join(ax[i + f], ax[ax_to_share])
        i += 1
    ax[1].legend(fontsize='x-small')
    ax[0].set_ylabel('Calcium events')
    # except:
    #     print('could not plot GLM kernels for', experience_level)
    if save_dir:
        title_string = 'oeid_' + str(ophys_experiment_id) + '_csid_' + str(cell_specimen_id) + '_' + cre_line
        fig.suptitle(title_string, x=0.5, y=1.15)
        utils.save_figure(fig, figsize, save_dir, 'kernel_example_plots', title_string + '_other_kernels')




######## coding scores plots ###########

def plot_coding_scores_for_cell(cell_specimen_id, ophys_experiment_id, results_pivoted, save_dir=None):
    '''
    Creates barplot of coding scores for a single cell in a single experiment and saves it
    '''

    identifier = str(ophys_experiment_id)+'_'+str(cell_specimen_id)
    # get dropouts just for one cell
    cell_dropouts = results_pivoted[results_pivoted.identifier == identifier]

        # which features to plot
    coding_score_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'all-images',
        'omissions', 'hits', 'misses', 'task', 'running', 'pupil', 'licks', 'behavioral']

    figsize = (3,6)
    fig, ax = plt.subplots(figsize=figsize)
    # get dropouts just for one cell
    ax = sns.barplot(data=np.abs(cell_dropouts[coding_score_features]), orient='h', color='gray', ax=ax) #color=sns.color_palette('Blues_r')[0], ax=ax)
    ax.set_xlabel('Coding score')
    ax.set_title('var_exp_full = '+str(np.round(cell_dropouts.variance_explained_full.values[0], 3)))

    if save_dir:
        cre_line = utils.get_abbreviated_cell_type(cell_dropouts.cre_line.values[0])
        title_string = 'oeid_'+str(ophys_experiment_id)+'_csid_'+str(cell_specimen_id)+'_'+cre_line
        fig.suptitle(title_string, x=0.5, y=1., fontsize=14)
        utils.save_figure(fig, figsize, save_dir, 'kernel_example_plots', title_string+'_coding_scores')






if __main__:

    import visual_behavior_glm.GLM_fit_dev as gfd
    import visual_behavior_glm.GLM_visualization_tools as gvt
    import visual_behavior_glm.GLM_analysis_tools as gat

    import visual_behavior.visualization.ophys.platform_paper_figures as ppf

    ### load metadata tables for platform experiments
    platform_experiments = loading.get_platform_paper_experiment_table(limit_to_closest_active=True)
    cells_table = loading.get_cell_table(platform_paper_only=True)

    print(len(platform_experiments)) # should be 402

    # save directory
    save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_plots\figure_3'

    ### load GLM results for platform paper experiments

    # Define model version
    glm_version = '24_events_all_L2_optimize_by_session'

    # Load GLM data
    run_params, results, results_pivoted, weights_df = gfd.load_analysis_dfs(glm_version)

    # limit to platform expts
    results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(platform_experiments.index.values)]

    # weights limited to platform expts
    weights_df = weights_df[weights_df.ophys_experiment_id.isin(platform_experiments.index.values)]


    ### Plot kernel activations for specfic experiment

    ophys_experiment_id = 808621034
    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    start_time = 2318  # 1500, 1450, 2550, 2320
    duration_seconds = 19.5
    plot_behavior_and_physio_timeseries_GLM(dataset, start_time, duration_seconds, save_dir=save_dir)


    ### Plot kernels and coding scores for a specific cell
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