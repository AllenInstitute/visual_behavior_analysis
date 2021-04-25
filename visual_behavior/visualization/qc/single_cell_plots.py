import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from visual_behavior.data_access import loading as data_loading
from visual_behavior.visualization import utils as utils
import visual_behavior.visualization.ophys.summary_figures as sf

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.response_processing as rp
import visual_behavior.ophys.response_analysis.utilities as ut


def plot_across_session_responses(ophys_container_id, cell_specimen_id, use_events=False, save_figure=True):
    """
    Generates plots characterizing single cell activity in response to stimulus, omissions, and changes.
    Compares across all sessions in a container for each cell, including the ROI mask across days.
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """
    experiments_table = data_loading.get_filtered_ophys_experiment_table()
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    expts = np.sort(container_expts.index.values)
    if use_events:
        ylabel = 'response'
        suffix = '_events'
    else:
        ylabel = 'dF/F'
        suffix = ''

    n = len(expts)
    figsize = (25, 20)
    fig, ax = plt.subplots(6, n, figsize=figsize)
    ax = ax.ravel()

    for i, ophys_experiment_id in enumerate(expts):
        try:

            dataset = data_loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False)
            print(dataset.dff_traces.index.values[0])
            if cell_specimen_id in dataset.dff_traces.index:
                analysis = ResponseAnalysis(dataset, use_events=use_events, use_extended_stimulus_presentations=False)
                sdf = ut.get_mean_df(analysis.get_response_df(df_name='stimulus_response_df'), analysis=analysis,
                                     conditions=['cell_specimen_id', 'is_change', 'image_name'], flashes=True, omitted=False,
                                     get_reliability=False, get_pref_stim=True, exclude_omitted_from_pref_stim=True)
                odf = ut.get_mean_df(analysis.get_response_df(df_name='omission_response_df'), analysis=analysis,
                                     conditions=['cell_specimen_id'], flashes=False, omitted=True,
                                     get_reliability=False, get_pref_stim=False, exclude_omitted_from_pref_stim=False)
                tdf = ut.get_mean_df(analysis.get_response_df(df_name='trials_response_df'), analysis=analysis,
                                     conditions=['cell_specimen_id', 'go', 'hit', 'change_image_name'], flashes=False, omitted=False,
                                     get_reliability=False, get_pref_stim=True, exclude_omitted_from_pref_stim=True)

                ct = dataset.cell_specimen_table.copy()
                cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id
                ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                          spacex=20, spacey=20, show_mask=True, ax=ax[i])
                ax[i].set_title(container_expts.loc[ophys_experiment_id].session_type[6:])

                colors = sns.color_palette('hls', 8) + [(0.5, 0.5, 0.5)]

                window = rp.get_default_stimulus_response_params()["window_around_timepoint_seconds"]
                cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == False)]
                for c, image_name in enumerate(np.sort(cell_data.image_name.unique())):
                    ax[i + n] = sf.plot_mean_trace_from_mean_df(cell_data[cell_data.image_name == image_name],
                                                                frame_rate=analysis.ophys_frame_rate, ylabel=ylabel,
                                                                legend_label=image_name, color=colors[c], interval_sec=0.5,
                                                                xlims=window, ax=ax[i + n])
                ax[i + n] = sf.plot_flashes_on_trace(ax[i + n], analysis, window=window, trial_type=None, omitted=False, alpha=0.15, facecolor='gray')
                ax[i + n].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n image response')

                analysis = ResponseAnalysis(dataset, use_events=False, use_extended_stimulus_presentations=False)
                tmp = analysis.get_response_df(df_name='stimulus_response_df')
                tmp['running'] = [True if run_speed > 2 else False for run_speed in tmp.mean_running_speed.values]
                sdf = ut.get_mean_df(tmp, analysis=analysis,
                                     conditions=['cell_specimen_id', 'is_change', 'image_name', 'running'], flashes=True, omitted=False,
                                     get_reliability=False, get_pref_stim=True, exclude_omitted_from_pref_stim=False)

                cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == False) & (sdf.pref_stim == True)]
                run_colors = [sns.color_palette()[3], sns.color_palette()[2]]
                for c, running in enumerate(np.sort(cell_data.running.unique())):
                    if len(cell_data[cell_data.running == running]) > 0:
                        ax[i + (n * 2)] = sf.plot_mean_trace_from_mean_df(cell_data[cell_data.running == running],
                                                                          frame_rate=analysis.ophys_frame_rate, ylabel=ylabel,
                                                                          legend_label=running, color=run_colors[c], interval_sec=0.5,
                                                                          xlims=window, ax=ax[i + (n * 2)])
                ax[i + (n * 2)].legend(fontsize='xx-small', title='running', title_fontsize='xx-small')
                ax[i + (n * 2)] = sf.plot_flashes_on_trace(ax[i + (n * 2)], analysis, window=window, trial_type=None, omitted=False, alpha=0.15, facecolor='gray')
                ax[i + (n * 2)].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n image response')

                window = rp.get_default_omission_response_params()["window_around_timepoint_seconds"]
                cell_data = odf[(odf.cell_specimen_id == cell_specimen_id)]
                ax[i + (n * 3)] = sf.plot_mean_trace_from_mean_df(cell_data,
                                                                  frame_rate=analysis.ophys_frame_rate, ylabel=ylabel,
                                                                  legend_label=image_name, color='gray', interval_sec=1,
                                                                  xlims=window, ax=ax[i + (n * 3)])
                ax[i + (n * 3)] = sf.plot_flashes_on_trace(ax[i + (n * 3)], analysis, window=window, trial_type=None, omitted=True, alpha=0.15, facecolor='gray')
                ax[i + (n * 3)].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n omission response')

                window = rp.get_default_trial_response_params()["window_around_timepoint_seconds"]
                cell_data = tdf[(tdf.cell_specimen_id == cell_specimen_id) & (tdf.go == True) & (tdf.pref_stim == True)]
                hit_colors = [sns.color_palette()[2], sns.color_palette()[3]]
                for c, hit in enumerate([True, False]):
                    if len(cell_data[cell_data.hit == hit]) > 0:
                        ax[i + (n * 4)] = sf.plot_mean_trace_from_mean_df(cell_data[cell_data.hit == hit],
                                                                          frame_rate=analysis.ophys_frame_rate, ylabel=ylabel,
                                                                          legend_label=hit, color=hit_colors[c], interval_sec=1,
                                                                          xlims=window, ax=ax[i + (n * 4)])
                ax[i + (n * 4)].legend(fontsize='xx-small', title='hit', title_fontsize='xx-small')
                ax[i + (n * 4)] = sf.plot_flashes_on_trace(ax[i + (n * 4)], analysis, window=window, trial_type='go', omitted=False, alpha=0.15, facecolor='gray')
                ax[i + (n * 4)].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n change response')

                fig.tight_layout()
                fig.suptitle(str(cell_specimen_id) + '_' + dataset.metadata_string, x=0.5, y=1.01,
                             horizontalalignment='center')
        except Exception as e:
            print('problem for cell_specimen_id:', cell_specimen_id)
            print(e)
    if save_figure:
        save_dir = utils.get_single_cell_plots_dir()
        utils.save_figure(fig, figsize, save_dir, 'across_session_responses', str(
            cell_specimen_id) + '_' + dataset.metadata_string + '_across_session_responses' + suffix)
        plt.close()


def plot_single_cell_activity_and_behavior(dataset, cell_specimen_id, save_figure=True):
    """
    Plots the full dFF trace for a cell, along with licking behavior, rewards, running speed, pupil area, and face motion.
    Useful to visualize whether the dFF trace tracks the behavior variables
    """
    figsize = (20, 10)
    fig, ax = plt.subplots(5, 1, figsize=figsize, sharex=True)
    colors = sns.color_palette()

    trace_timestamps = dataset.ophys_timestamps
    trace = dataset.dff_traces.loc[cell_specimen_id].dff
    ax[0].plot(trace_timestamps, trace, label='mean_trace', color=colors[0])
    ax[0].set_ylabel('dF/F')

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    ax[1].plot(lick_timestamps, licks, '|', label='licks', color=colors[3])
    ax[1].set_ylabel('licks')
    ax[1].set_yticklabels([])

    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values
    ax[2].plot(running_timestamps, running_speed, label='running_speed', color=colors[4])
    ax[2].set_ylabel('run speed\n(cm/s)')

    try:
        pupil_area = dataset.eye_tracking.pupil_area.values
        pupil_timestamps = dataset.eye_tracking.timestamps.values
        ax[3].plot(pupil_timestamps, pupil_area, label='pupil_area', color=colors[9])
    except Exception:
        print('no pupil for', dataset.ophys_experiment_id)
    ax[3].set_ylabel('pupil area\n pixels**2')
    ax[3].set_ylim(-50, 30000)

    try:
        face_motion = dataset.behavior_movie_pc_activations[:, 0]
        face_timestamps = dataset.timestamps['eye_tracking'].timestamps
        ax[4].plot(face_timestamps, face_motion, label='face_motion_PC0', color=colors[2])
    except Exception:
        print('no face motion for', dataset.ophys_experiment_id)
    ax[4].set_ylabel('face motion\n PC0 activation')

    for x in range(5):
        ax[x].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)
    ax[4].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    #     ax[x].legend(loc='upper left', fontsize='x-small')
    plt.subplots_adjust(wspace=0, hspace=0.1)
    ax[0].set_title(str(cell_specimen_id) + '_' + dataset.metadata_string)
    if save_figure:
        utils.save_figure(fig, figsize, utils.get_single_cell_plots_dir(), 'dff_trace_and_behavior',
                          str(cell_specimen_id) + '_' + dataset.metadata_string + '_dff_trace_and_behavior')
        plt.close()
