import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.processing as processing
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.visualization.utils as utils


def plot_behavior(ophys_experiment_id, xlim_seconds=None, plot_stimuli=True, ax=None):
    """
    Plots licking behavior, rewards, running speed, pupil area, and face motion.
    Useful to visualize whether the overal activity tracks the behavior variables
    """

    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    if xlim_seconds is None:
        xlim_seconds = [dataset.stimulus_timestamps[0], dataset.stimulus_timestamps[-1]]

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    licks[:] = -2

    reward_timestamps = dataset.rewards.timestamps.values
    rewards = np.zeros(len(reward_timestamps))
    rewards[:] = -4

    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values

    pupil_area = dataset.eye_tracking.pupil_area.values
    pupil_timestamps = dataset.eye_tracking.timestamps.values

    if ax is None:
        figsize = (10, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = sns.color_palette()

    ln0 = ax.plot(lick_timestamps, licks, '|', label='licks', color=colors[3])
    ln1 = ax.plot(reward_timestamps, rewards, 'o', label='rewards', color=colors[9])

    ln2 = ax.plot(running_timestamps, running_speed, label='running_speed', color=colors[2], zorder=100)
    ax.set_ylabel('run speed\n(cm/s)')
    ax.set_ylim(ymin=-8)

    ax2 = ax.twinx()
    ln3 = ax2.plot(pupil_timestamps, np.sqrt(pupil_area), label='pupil_area', color=colors[4], zorder=0)
    #     ln4 = ax2.plot(pupil_timestamps, dataset.eye_tracking.eye_center_x.values/5., label='eye_center_x', color=colors[1], zorder=2)

    ax2.set_ylabel('pupil area\n(pixels)')
    #     ax2.set_ylim(0, 200)

    axes_to_label = ln0 + ln1 + ln2 + ln3  # +ln4
    labels = [label.get_label() for label in axes_to_label]
    ax.legend(axes_to_label, labels, loc='upper left', fontsize='x-small')

    #     try:
    #         face_motion = dataset.behavior_movie_pc_activations[:, 0]
    #         face_timestamps = dataset.behavior_movie_timestamps
    #         ax2.plot(face_timestamps, face_motion, label='face_motion_PC0', color=colors[8], zorder=0)
    # #         ax.set_ylabel('face motion\n PC0 activation')
    #     except Exception as e:
    #         print(dataset.ophys_experiment_id)
    #         print(e)

    if plot_stimuli:
        ax = sf.add_stim_color_span(dataset, ax, xlim=xlim_seconds)

    ax.set_xlim(xlim_seconds)
    #     ax.tick_params(which='both', bottom=False, top=False, right=False, left=True,
    #                     labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    #     ax.set_xlabel('time (sec)')
    return ax


def plot_traces(dataset, include_cell_traces=True, plot_stimuli=True, xlim_seconds=None, use_events=False,
                label='population average', color=None, ax=None):
    colors = sns.color_palette()
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 3))
    if xlim_seconds is None:
        xlim_seconds = [dataset.ophys_timestamps[0], dataset.ophys_timestamps[-1]]
    if use_events:
        if color is None:
            color = colors[3]
        multiplier = 1000
        column = 'filtered_events'
        dff_traces = dataset.events.copy()
    else:
        if color is None:
            color = colors[0]
        multiplier = 100
        column = 'dff'
        dff_traces = dataset.dff_traces.copy()
    dff_traces = processing.compute_robust_snr_on_dataframe(dff_traces)
    highest_snr = np.argsort(dff_traces[dff_traces.robust_snr.isnull() == False].robust_snr.values)[-10:]
    highest_cells = dff_traces[dff_traces.robust_snr.isnull() == False].index.values[highest_snr]
    ax.plot(dataset.ophys_timestamps, dff_traces[column].mean() * multiplier, color=color, zorder=100, label=label)
    if include_cell_traces:
        for cell_specimen_id in highest_cells:
            ax.plot(dataset.ophys_timestamps, dff_traces.loc[cell_specimen_id][column] * multiplier, label=str(cell_specimen_id))  # , color='gray')
        ax.plot(dataset.ophys_timestamps, dff_traces.loc[cell_specimen_id][column] * multiplier)  # color='gray',
    ax.set_ylabel('dF/F')
    ax.set_xlabel('time (sec)')
    ax.legend(fontsize='xx-small', loc='upper left')
    if plot_stimuli:
        ax = sf.add_stim_color_span(dataset, ax, xlim=xlim_seconds)
    ax.set_xlim(xlim_seconds)
#     ax.tick_params(which='both', bottom=False, top=False, right=False, left=True,
#                     labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    return ax


def plot_behavior_model_weights(dataset, xlim_seconds=None, plot_stimuli=True, ax=None):
    if xlim_seconds is None:
        xlim_seconds = [dataset.stimulus_timestamps[0], dataset.stimulus_timestamps[-1]]
    st = dataset.stimulus_presentations.copy()
    st = loading.add_model_outputs_to_stimulus_presentations(st, dataset.metadata['behavior_session_id'])
    colors = sns.color_palette()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(st.start_time.values, st.bias.values, color=colors[2], label='bias')
    ax.plot(st.start_time.values, st.task0.values, color=colors[0], label='task')
    ax.plot(st.start_time.values, st.timing1D.values, color=colors[1], label='timing')
    ax.plot(st.start_time.values, st.omissions1.values, color=colors[4], label='post-omission')
    ax.legend(fontsize='x-small', loc='upper left')
    if plot_stimuli:
        ax = sf.add_stim_color_span(dataset, ax, xlim=xlim_seconds)
    ax.set_xlim(xlim_seconds)
    return ax


def plot_behavior_and_cell_traces_pop_avg(dataset, xlim_seconds=None, save_figure=True):
    if xlim_seconds is None:
        suffix = ''
    elif xlim_seconds[1] < 2000:
        suffix = 'early'
    elif xlim_seconds[1] > -2000:
        suffix = 'late'

    figsize = (15, 8)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    try:
        ax[0] = plot_behavior_model_weights(dataset, xlim_seconds=xlim_seconds, plot_stimuli=True, ax=ax[0])
    except BaseException:
        print('no behavior model output for', dataset.ophys_experiment_id)
    ax[1] = plot_behavior(dataset.ophys_experiment_id, xlim_seconds=xlim_seconds, plot_stimuli=True, ax=ax[1])
    ax[1] = plot_traces(dataset, include_cell_traces=False, plot_stimuli=False, xlim_seconds=xlim_seconds, label='pop. avg. dFF', ax=ax[1])
    ax[1] = plot_traces(dataset, include_cell_traces=False, plot_stimuli=False, xlim_seconds=xlim_seconds, label='pop. avg. events',
                        use_events=True, ax=ax[1])
    ax[2] = plot_traces(dataset, include_cell_traces=True, plot_stimuli=True, xlim_seconds=xlim_seconds, ax=ax[2])
    plt.subplots_adjust(wspace=0, hspace=0.1)
    ax[0].set_title(dataset.metadata_string)

    if save_figure:
        save_dir = os.path.abspath(os.path.join(loading.get_qc_plots_dir(), 'timeseries_plots'))
        utils.save_figure(fig, figsize, save_dir, 'behavior_traces_cell_traces_pop_avg',
                          dataset.metadata_string + '_' + suffix)


def plot_behavior_and_pop_avg(dataset, xlim_seconds=None, save_figure=True):
    if xlim_seconds is None:
        suffix = ''
    elif xlim_seconds[1] < 2000:
        suffix = 'early'
    elif xlim_seconds[1] > -2000:
        suffix = 'late'

    figsize = (15, 8)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    try:
        ax[0] = plot_behavior_model_weights(dataset, xlim_seconds=xlim_seconds, plot_stimuli=True, ax=ax[0])
    except BaseException:
        print('no behavior model output for', dataset.ophys_experiment_id)
    ax[1] = plot_behavior(dataset.ophys_experiment_id, xlim_seconds=xlim_seconds, plot_stimuli=True, ax=ax[1])
    ax[2] = plot_traces(dataset, include_cell_traces=False, plot_stimuli=True, xlim_seconds=xlim_seconds, ax=ax[2])
    plt.subplots_adjust(wspace=0, hspace=0.1)
    ax[0].set_title(dataset.metadata_string)

    if save_figure:
        save_dir = os.path.abspath(os.path.join(loading.get_qc_plots_dir(), 'timeseries_plots'))
        utils.save_figure(fig, figsize, save_dir, 'behavior_traces_population_average',
                          dataset.metadata_string + '_' + suffix)


def plot_behavior_and_pop_avg_mesoscope(ophys_session_id, xlim_seconds=None, save_figure=True):
    if xlim_seconds is None:
        suffix = ''
    elif xlim_seconds[1] < 2000:
        suffix = 'early'
    elif xlim_seconds[1] > -2000:
        suffix = 'late'

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiment_ids = experiments_table[experiments_table.ophys_session_id == ophys_session_id].sort_values(by='date_of_acquisition').index.values
    experiment_id = experiment_ids[0]

    dataset = loading.get_ophys_dataset(experiment_id)

    colors = sns.color_palette()
    figsize = (15, 8)
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    try:
        ax[0] = plot_behavior_model_weights(dataset, xlim_seconds=xlim_seconds, plot_stimuli=True, ax=ax[0])
    except BaseException:
        print('no behavior model output for', dataset.ophys_experiment_id)
    ax[1] = plot_behavior(dataset.ophys_experiment_id, xlim_seconds=xlim_seconds, plot_stimuli=True, ax=ax[1])

    label = dataset.metadata['targeted_structure'] + ', ' + str(dataset.metadata['imaging_depth'])
    ax[2] = plot_traces(dataset, include_cell_traces=False, plot_stimuli=True, xlim_seconds=xlim_seconds, label=label, color=colors[0], ax=ax[2])
    for i, experiment_id in enumerate(experiment_ids[1:]):
        dataset = loading.get_ophys_dataset(experiment_id)
        label = dataset.metadata['targeted_structure'] + ', ' + str(dataset.metadata['imaging_depth'])
        ax[2] = plot_traces(dataset, include_cell_traces=False, plot_stimuli=False, xlim_seconds=xlim_seconds, label=label, color=colors[i + 1], ax=ax[2])
    ax[2].legend(fontsize='x-small', loc='upper left')
    plt.subplots_adjust(wspace=0, hspace=0.1)
    ax[0].set_title(dataset.metadata_string)

    if save_figure:
        save_dir = os.path.abspath(os.path.join(loading.get_qc_plots_dir(), 'timeseries_plots'))
        utils.save_figure(fig, figsize, save_dir, 'behavior_traces_population_average_mesoscope',
                          dataset.metadata_string + '_' + suffix)
