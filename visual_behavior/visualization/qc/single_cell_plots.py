import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from visual_behavior.data_access import loading as loading
from visual_behavior.data_access import utilities as utilities

import visual_behavior.visualization.ophys.summary_figures as sf

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.response_processing as rp
import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.visualization.utils as utils



def plot_across_session_responses(ophys_container_id, cell_specimen_id, use_events=False, save_figure=True):
    """
    Generates plots characterizing single cell activity in response to stimulus, omissions, and changes.
    Compares across all sessions in a container for each cell, including the ROI mask across days.
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """
    import visual_behavior.data_access.utilities as utilities
    experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    expts = np.sort(container_expts.index.values)
    if use_events:
        ylabel = 'response'
        suffix = '_events'
    else:
        ylabel = 'dF/F'
        suffix = ''
    window = [-0.5, 1.5]
    interpolate = True
    output_sampling_rate = 30

    n = len(expts)
    figsize = (25, 20)
    fig, ax = plt.subplots(6, n, figsize=figsize)
    ax = ax.ravel()
    print('ophys_container_id:', ophys_container_id)
    for i, ophys_experiment_id in enumerate(expts):
        print('ophys_experiment_id:', ophys_experiment_id)
        try:
            dataset = loading.get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False)

            if cell_specimen_id in dataset.dff_traces.index:
                sdf = loading.get_stimulus_response_df(dataset, data_type='dff', event_type='all',
                                                       time_window=window, interpolate=interpolate, output_sampling_rate=output_sampling_rate)
                cdf = ut.get_mean_df(sdf, conditions=['cell_specimen_id', 'is_change', 'image_name'])
                odf = ut.get_mean_df(sdf[sdf.omitted==True], conditions=['cell_specimen_id'])
                tdf = ut.get_mean_df(sdf[sdf.is_change==True], conditions=['cell_specimen_id', 'go', 'hit', 'change_image_name'])
                timestamps = sdf.trace_timestamps.values[0]

                ct = dataset.cell_specimen_table.copy()
                cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id
                ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                          spacex=20, spacey=20, show_mask=True, ax=ax[i])
                ax[i].set_title(container_expts.loc[ophys_experiment_id].session_type[6:])

                colors = sns.color_palette('hls', 8) + [(0.5, 0.5, 0.5)]

                # average image response each image
                cell_data = cdf[(cdf.cell_specimen_id == cell_specimen_id) & (cdf.is_change == False)]
                for c, image_name in enumerate(np.sort(cell_data.image_name.unique())):
                    ax[i + n] = utils.plot_mean_trace_from_mean_df(cell_data[cell_data.image_name == image_name],
                                                                frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                legend_label=image_name, color=colors[c], interval_sec=0.5,
                                                                xlims=window, ax=ax[i + n])
                ax[i + n] = utils.plot_flashes_on_trace(ax[i + n], timestamps, change=True, omitted=False, alpha=0.15, facecolor='gray')
                ax[i + n].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n image response')

                # running vs not-running
                try:
                    tmp = sdf.copy()
                    tmp['running'] = [True if run_speed > 2 else False for run_speed in tmp.mean_running_speed.values]
                    sdf = ut.get_mean_df(tmp, analysis=analysis,
                                         conditions=['cell_specimen_id', 'is_change', 'image_name', 'running'], flashes=True, omitted=False,
                                         get_pref_stim=True, exclude_omitted_from_pref_stim=False)

                    cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == False) & (sdf.pref_stim == True)]
                    run_colors = [sns.color_palette()[3], sns.color_palette()[2]]
                    for c, running in enumerate(np.sort(cell_data.running.unique())):
                        if len(cell_data[cell_data.running == running]) > 0:
                            ax[i + (n * 2)] = utils.plot_mean_trace_from_mean_df(cell_data[cell_data.running == running],
                                                                              frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                              legend_label=running, color=run_colors[c], interval_sec=0.5,
                                                                              xlims=window, ax=ax[i + (n * 2)])
                    ax[i + (n * 2)].legend(fontsize='xx-small', title='running', title_fontsize='xx-small')
                    ax[i + (n * 2)] = utils.plot_flashes_on_trace(ax[i + (n * 2)], timestamps, change=True, omitted=False, alpha=0.15, facecolor='gray')
                    ax[i + (n * 2)].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n image response')
                except:
                    print('couldnt plot running / not-running panel')

                # omissions
                cell_data = odf[(odf.cell_specimen_id == cell_specimen_id)]
                ax[i + (n * 3)] = utils.plot_mean_trace_from_mean_df(cell_data,
                                                                  frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                  legend_label=image_name, color='gray', interval_sec=1,
                                                                  xlims=window, ax=ax[i + (n * 3)])
                ax[i + (n * 3)] = utils.plot_flashes_on_trace(ax[i + (n * 3)], timstamps, change=False, omitted=True, alpha=0.15, facecolor='gray')
                ax[i + (n * 3)].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n omission response')


                # hit miss
                cell_data = tdf[(tdf.cell_specimen_id == cell_specimen_id) & (tdf.go == True) & (tdf.pref_stim == True)]
                hit_colors = [sns.color_palette()[2], sns.color_palette()[3]]
                for c, hit in enumerate([True, False]):
                    if len(cell_data[cell_data.hit == hit]) > 0:
                        ax[i + (n * 4)] = utils.plot_mean_trace_from_mean_df(cell_data[cell_data.hit == hit],
                                                                          frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                          legend_label=hit, color=hit_colors[c], interval_sec=1,
                                                                          xlims=window, ax=ax[i + (n * 4)])
                ax[i + (n * 4)].legend(fontsize='xx-small', title='hit', title_fontsize='xx-small')
                ax[i + (n * 4)] = utils.plot_flashes_on_trace(ax[i + (n * 4)], timestamps, change=True, omitted=False, alpha=0.15, facecolor='gray')
                ax[i + (n * 4)].set_title(container_expts.loc[ophys_experiment_id].session_type[6:] + '\n change response')

                fig.tight_layout()
                fig.suptitle(str(cell_specimen_id) + '_' + dataset.metadata_string, x=0.5, y=1.01,
                             horizontalalignment='center')
        except Exception as e:
            print('problem for cell_specimen_id:', cell_specimen_id, ', ophys_experiment_id:', ophys_experiment_id)
            print(e)
    if save_figure:
        save_dir = utils.get_single_cell_plots_dir()
        utils.save_figure(fig, figsize, save_dir, 'across_session_responses', str(
            cell_specimen_id) + '_' + dataset.metadata_string + '_across_session_responses' + suffix)
        plt.close()


def plot_across_session_responses_from_dataset_dict(data_dict, ophys_container_id, cell_specimen_id, experiments_table,
                                                    data_type='dff', save_figure=True):
    """
    Generates plots characterizing single cell activity across sessions, in response to stimulus, omissions, and changes.
    Compares across all sessions in a container for each cell, including the ROI mask across days.
    Additional panels include response for each image, change response for hit vs miss, and change response for running vs. not running
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """

    # get info for this container
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by=['date_of_acquisition'])
    ophys_experiment_ids = container_expts.index.values
    print('mouse_id:', container_expts.mouse_id.unique())
    print('ophys_container_id:', ophys_container_id)
    print('there are', len(ophys_experiment_ids), 'in this container')
    suffix = '_'+data_type
    if data_type == 'dff':
        ylabel = 'dF/F'
    else:
        ylabel = 'response'

    window = [-0.5, 1.5]
    n = len(ophys_experiment_ids)
    figsize = (4*n, 20)
    fig, ax = plt.subplots(5, n, figsize=figsize, sharey='row')
    ax = ax.ravel()
    print('ophys_container_id:', ophys_container_id)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('ophys_experiment_id:', ophys_experiment_id)
        try:
            dataset = data_dict[ophys_experiment_id]['dataset']['dataset']

            # if the selected cell is not in this session (i.e. is not matched), plot it, otherwise skip
            if cell_specimen_id in dataset.dff_traces.index:
                print(cell_specimen_id, 'is in', ophys_experiment_id)
                # get stimulus response df for this experiment (response df should have been created using event_type='all')
                sdf = data_dict[ophys_experiment_id][data_type]['stimulus_response_df'].copy()
                output_sampling_rate = sdf.output_sampling_rate.values[0]
                response_window_duration = sdf.response_window_duration.values[0]
                timestamps = sdf.trace_timestamps.values[0]
                window = [timestamps[0], timestamps[-1]]

                # plot cell mask
                print('plotting ROI mask')
                ct = dataset.cell_specimen_table.copy()
                cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id
                ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.average_projection, cell_roi_id,
                                          spacex=50, spacey=50, show_mask=True, ax=ax[i])

                session_type = container_expts.loc[ophys_experiment_id].session_type
                s = session_type.split('_')
                session_type = s[0] + '_' + s[1] + '_' + s[2]
                ax[i].set_title(str(container_expts.loc[ophys_experiment_id].date_of_acquisition)[:10]+'\n'+session_type)

                # plot average response for each image (for all non-change image presentations)
                # get trial averaged responses for various conditions
                print('plotting image responses')
                cdf = ut.get_mean_df(sdf, conditions=['cell_specimen_id', 'is_change', 'image_name'],
                                     frame_rate=output_sampling_rate, time_window=window,
                                     response_window_duration=response_window_duration,
                                     get_pref_stim=True)
                colors = sns.color_palette()
                cell_data = cdf[(cdf.cell_specimen_id == cell_specimen_id) & (cdf.is_change == True)]
                for c, image_name in enumerate(np.sort(cell_data.image_name.unique())):
                    ax[i + n] = utils.plot_mean_trace_from_mean_df(cell_data[cell_data.image_name == image_name],
                                                                frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                legend_label=image_name, color=colors[c], interval_sec=0.5,
                                                                xlims=[-1, 2], ax=ax[i + n])
                    ax[i + n].legend(loc='upper right', fontsize='xx-small')
                ax[i + n] = utils.plot_flashes_on_trace(ax[i + n], timestamps, change=True, omitted=False, alpha=0.15, facecolor='gray')
                ax[i + n].set_title('oeid: '+str(ophys_experiment_id))

                # plot mean omission response
                print('plotting omissions')
                try:
                    odf = ut.get_mean_df(sdf[sdf.omitted==True], conditions=['cell_specimen_id'],
                                         frame_rate=output_sampling_rate, time_window=window,
                                         response_window_duration=response_window_duration,
                                         get_pref_stim=False)
                    cell_data = odf[(odf.cell_specimen_id == cell_specimen_id)]
                    ax[i + (n * 2)] = utils.plot_mean_trace_from_mean_df(cell_data,
                                                                      frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                      legend_label=image_name, color='gray', interval_sec=1,
                                                                      xlims=[-1, 2], ax=ax[i + (n * 2)])
                    ax[i + (n * 2)] = utils.plot_flashes_on_trace(ax[i + (n * 2)], timstamps, change=False, omitted=True, alpha=0.15, facecolor='gray')
                    ax[i + (n * 2)].set_title('osid: '+str(container_expts.loc[ophys_experiment_id].ophys_session_id))
                except:
                    print('couldnt plot omissions')


                # plot mean stimulus response for running vs not-running
                print('plotting running not running')
                try:
                    tmp = sdf.copy()
                    # create boolean column indicating running vs. not running, using 2cm/sec as threshold
                    tmp['running'] = [True if run_speed > 2 else False for run_speed in tmp.mean_running_speed.values]
                    rdf = ut.get_mean_df(tmp, conditions=['cell_specimen_id', 'is_change', 'image_name', 'running'],
                                         frame_rate=output_sampling_rate, time_window=window,
                                         response_window_duration=response_window_duration,
                                         get_pref_stim=True)
                    # get responses for non-changes with preferred stimulus
                    cell_data = rdf[(rdf.cell_specimen_id == cell_specimen_id) & (rdf.is_change == False) & (rdf.pref_stim == True)]
                    run_colors = [sns.color_palette()[3], sns.color_palette()[2]]
                    # loop through running conditions and plot
                    for c, running in enumerate(np.sort(cell_data.running.unique())):
                        if len(cell_data[cell_data.running == running]) > 0: # only plot if there is data for this condition
                            ax[i + (n * 3)] = utils.plot_mean_trace_from_mean_df(cell_data[cell_data.running == running],
                                                                              frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                              legend_label=running, color=run_colors[c], interval_sec=0.5,
                                                                              xlims=[-1, 2], ax=ax[i + (n * 3)])
                    ax[i + (n * 3)].legend(fontsize='xx-small', title='running', title_fontsize='xx-small')
                    ax[i + (n * 3)] = utils.plot_flashes_on_trace(ax[i + (n * 3)], timestamps, change=True, omitted=False, alpha=0.15, facecolor='gray')
                    ax[i + (n * 3)].set_title('running vs stationary')
                except:
                    print('couldnt plot running / not-running panel')


                # plot change repsonse for hit vs. miss for pref image
                print('plotting hit vs. miss')
                try:
                    tmp = sdf.copy()
                    tdf = ut.get_mean_df(tmp[tmp.is_change==True], conditions=['cell_specimen_id', 'go', 'hit'],
                                         frame_rate=output_sampling_rate, time_window=window,
                                         response_window_duration=response_window_duration,
                                         get_pref_stim=False)
                    cell_data = tdf[(tdf.cell_specimen_id == cell_specimen_id) & (tdf.go == True)]
                    hit_colors = [sns.color_palette()[2], sns.color_palette()[3]]
                    for c, hit in enumerate([True, False]):
                        if len(cell_data[cell_data.hit == hit]) > 0:
                            ax[i + (n * 4)] = utils.plot_mean_trace_from_mean_df(cell_data[cell_data.hit == hit],
                                                                              frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                              legend_label=hit, color=hit_colors[c], interval_sec=1,
                                                                              xlims=[-1, 2], ax=ax[i + (n * 4)])
                    ax[i + (n * 4)].legend(fontsize='xx-small', title='hit', title_fontsize='xx-small')
                    ax[i + (n * 4)] = utils.plot_flashes_on_trace(ax[i + (n * 4)], timestamps, change=True, omitted=False, alpha=0.15, facecolor='gray')
                    ax[i + (n * 4)].set_title('hit vs miss')
                except:
                    print('couldnt plot hit vs. miss')

        except Exception as e:
            print('problem for cell_specimen_id:', cell_specimen_id, ', ophys_experiment_id:', ophys_experiment_id)
            print(e)

    # overall plot title
    fig.tight_layout()
    metadata_string = utils.get_container_metadata_string(dataset.metadata)
    fig.suptitle(metadata_string + '_' + str(cell_specimen_id), x=0.5, y=1.01, horizontalalignment='center')
    if save_figure:
        save_dir = utils.get_single_cell_plots_dir()
        utils.save_figure(fig, figsize, save_dir, 'matched_cell_across_session_responses',
                          metadata_string + '_' + str(cell_specimen_id)  + suffix)




def plot_across_session_responses_from_multi_session_dfs(ophys_container_id, cell_specimen_id, experiments_table,
                                                        change_image_mdf, omission_mdf, output_sampling_rate=30,
                                                        data_type='dff', save_figure=True):
    """
    Generates plots characterizing single cell activity across sessions for matched cells.
    Plots the ROI mask across days, as well as the average change response for each image in the session,
    and the average omission response for sessions where omissions are present.
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """

    # get info for this container
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id].sort_values(by=['date_of_acquisition'])
    ophys_experiment_ids = container_expts.index.values
    print('mouse_id:', container_expts.mouse_id.unique())
    print('ophys_container_id:', ophys_container_id)
    print('there are', len(ophys_experiment_ids), 'experiments in this container')
    suffix = '_'+data_type
    if data_type == 'dff':
        ylabel = 'dF/F'
    else:
        ylabel = 'response'

    window = [-1, 2]

    n = len(ophys_experiment_ids)
    figsize = (4*n, 15)
    fig, ax = plt.subplots(3, n, figsize=figsize, sharey='row')
    ax = ax.ravel()
    print('ophys_container_id:', ophys_container_id)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('ophys_experiment_id:', ophys_experiment_id)
        try:
            dataset = loading.get_ophys_dataset(ophys_experiment_id)

            # create another plot while we have dataset loaded
            try:
                plot_single_cell_activity_and_behavior(dataset, cell_specimen_id, save_figure=True)
            except:
                print('couldnt plot single cell activity and behavior')

            # if the selected cell is not in this session (i.e. is not matched), plot it, otherwise skip
            if cell_specimen_id in dataset.dff_traces.index:
                print(cell_specimen_id, 'is in', ophys_experiment_id)

                # plot cell mask
                print('plotting ROI mask')
                ct = dataset.cell_specimen_table.copy()
                cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id
                ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.average_projection, cell_roi_id,
                                          spacex=50, spacey=50, show_mask=True, ax=ax[i])

                session_type = container_expts.loc[ophys_experiment_id].session_type
                s = session_type.split('_')
                session_type = s[0] + '_' + s[1] + '_' + s[2]
                ax[i].set_title(str(container_expts.loc[ophys_experiment_id].date_of_acquisition)[:10]+'\n'+session_type)

                # plot average response for each image (for all non-change image presentations)
                # get trial averaged responses for various conditions
                print('plotting image responses')
                cdf = change_image_mdf.copy()
                cdf = cdf[cdf.ophys_experiment_id==ophys_experiment_id]
                colors = sns.color_palette()
                cell_data = cdf[(cdf.cell_specimen_id == cell_specimen_id) & (cdf.is_change == True)]
                timestamps = cell_data.trace_timestamps.values[0]
                for c, image_name in enumerate(np.sort(cell_data.image_name.unique())):
                    ax[i + n] = utils.plot_mean_trace_from_mean_df(cell_data[cell_data.image_name == image_name],
                                                                frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                legend_label=image_name, color=colors[c], interval_sec=0.5,
                                                                xlims=[-1, 2], ax=ax[i + n])
                    ax[i + n].legend(loc='upper right', fontsize='xx-small')
                ax[i + n] = utils.plot_flashes_on_trace(ax[i + n], timestamps, change=True, omitted=False, alpha=0.15, facecolor='gray')
                ax[i + n].set_title('oeid: '+str(ophys_experiment_id))

                # plot mean omission response
                print('plotting omissions')
                try:
                    odf = omission_mdf.copy()
                    odf = odf[odf.ophys_experiment_id==ophys_experiment_id]
                    cell_data = odf[(odf.cell_specimen_id == cell_specimen_id)]
                    timestamps = cell_data.trace_timestamps.values[0]
                    ax[i + (n * 2)] = utils.plot_mean_trace_from_mean_df(cell_data,
                                                                      frame_rate=output_sampling_rate, ylabel=ylabel,
                                                                      legend_label=image_name, color='gray', interval_sec=1,
                                                                      xlims=[-1, 2], ax=ax[i + (n * 2)])
                    ax[i + (n * 2)] = utils.plot_flashes_on_trace(ax[i + (n * 2)], timstamps, change=False, omitted=True, alpha=0.15, facecolor='gray')
                    ax[i + (n * 2)].set_title('osid: '+str(container_expts.loc[ophys_experiment_id].ophys_session_id))
                except:
                    print('couldnt plot omissions')
        except:
            print('problem for', ophys_experiment_id)


    # overall plot title
    fig.tight_layout()
    metadata_string = utils.get_container_metadata_string(dataset.metadata)
    fig.suptitle(metadata_string + '_' + str(cell_specimen_id), x=0.5, y=1.01, horizontalalignment='center')
    if save_figure:
        save_dir = utils.get_single_cell_plots_dir()
        utils.save_figure(fig, figsize, save_dir, 'matched_cell_across_session_responses_from_mdf',
                          metadata_string + '_' + str(cell_specimen_id)  + suffix)



def plot_single_cell_activity_and_behavior(dataset, cell_specimen_id, save_figure=True):
    """
    Plots the full dFF trace for a cell, along with licking behavior, rewards, running speed, pupil area, and face motion.
    Useful to visualize whether the dFF trace tracks the behavior variables
    """
    figsize = (20, 12)
    fig, ax = plt.subplots(6, 1, figsize=figsize, sharex=True)
    colors = sns.color_palette()


    trace_timestamps = dataset.ophys_timestamps
    trace = dataset.corrected_fluorescence_traces.loc[cell_specimen_id].corrected_fluorescence
    ax[0].plot(trace_timestamps, trace, label='corrected fluorescence', color=colors[2])
    ax[0].set_ylabel('fluorescence')

    trace = dataset.dff_traces.loc[cell_specimen_id].dff
    events = dataset.events.loc[cell_specimen_id].events
    # events[events==0] = np.nan
    ax[1].plot(trace_timestamps, trace, label='dF/F', color=colors[0])
    ax[1].set_ylabel('dF/F')
    ax2 = ax[1].twinx()
    ax2.plot(trace_timestamps, events, label='events', color=colors[8])
    ax2.set_ylabel('events')
    ax[1].legend(fontsize='xx-small', loc='upper left')
    ax2.legend(fontsize='xx-small', loc='upper right')

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    ax[2].plot(lick_timestamps, licks, '|', label='licks', color=colors[3])
    ax[2].set_ylabel('licks')
    ax[2].set_yticklabels([])

    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values
    ax[3].plot(running_timestamps, running_speed, label='running_speed', color=colors[4])
    ax[3].set_ylabel('run speed\n(cm/s)')

    try:
        pupil_area = dataset.eye_tracking.pupil_area.values
        pupil_timestamps = dataset.eye_tracking.timestamps.values
        ax[4].plot(pupil_timestamps, pupil_area, label='pupil_area', color=colors[9])
    except Exception:
        print('no pupil for', dataset.ophys_experiment_id)
    ax[4].set_ylabel('pupil area\n pixels**2')
    ax[4].set_ylim(-50, 30000)

    try:
        face_motion = dataset.behavior_movie_pc_activations[:, 0]
        face_timestamps = dataset.timestamps['eye_tracking'].timestamps
        ax[5].plot(face_timestamps, face_motion, label='face_motion_PC0', color=colors[2])
    except Exception:
        print('no face motion for', dataset.ophys_experiment_id)
    ax[5].set_ylabel('face motion\n PC0 activation')

    for x in range(6):
        ax[x].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                          labelbottom=False, labeltop=False, labelright=False, labelleft=True)
    ax[5].tick_params(which='both', bottom=False, top=False, right=False, left=True,
                      labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    #     ax[x].legend(loc='upper left', fontsize='x-small')
    plt.subplots_adjust(wspace=0, hspace=0.1)
    metadata_string = utils.get_metadata_string(dataset.metadata)
    ax[0].set_title(metadata_string + '_' + str(cell_specimen_id))
    if save_figure:
        utils.save_figure(fig, figsize, utils.get_single_cell_plots_dir(), 'dff_trace_and_behavior',
                          metadata_string + '_' + str(cell_specimen_id) + '_dff_trace_and_behavior')
        plt.close()


def plot_cell_roi_mask_and_dff_trace(dataset, cell_roi_id, save_figure=True):
    dff_traces = dataset.dff_traces.copy()
    corrected_fluorescence = dataset.corrected_fluorescence_traces.copy()
    roi_masks = dataset.roi_masks.copy()
    max_projection = dataset.max_projection.data.copy()
    average_image = dataset.average_projection.data.copy()
    metadata = dataset.metadata.copy()

    figsize=(20,10)
    fig, ax = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'width_ratios':[1,3]})
    ax = ax.ravel()
    ax[0] = sf.plot_cell_zoom(roi_masks, average_image, cell_roi_id, spacex=50, spacey=50, show_mask=True, ax=ax[0])
    # create twinax
    ax1 = ax[1].twinx()
    # dff traces
    ax1.plot(dataset.ophys_timestamps, dff_traces[dff_traces.cell_roi_id == cell_roi_id].dff.values[0], label='dF/F', color='purple')
    ax1.set_xlim(500, 680)
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('dF/F')
    ax1.legend()
    # show corrected fluorescence as well
    ax[1].plot(dataset.ophys_timestamps, corrected_fluorescence[corrected_fluorescence.cell_roi_id==cell_roi_id].corrected_fluorescence.values[0], color='g', label='fluor')
    ax[1].set_ylabel('corrected fluorescence')
    ax[1].legend()


    ax[2] = sf.plot_cell_zoom(roi_masks, average_image, cell_roi_id, show_mask=True, alpha=1, full_image=True, ax=ax[2])
    # create twinx
    ax3 = ax[3].twinx()
    ax3.plot(dataset.ophys_timestamps, dff_traces[dff_traces.cell_roi_id==cell_roi_id].dff.values[0], label='dF/F', color='purple')
    ax3.set_xlabel('time (sec)')
    ax3.set_ylabel('dF/F')
    ax3.legend()
    # show corrected fluorescence as well
    ax[3].plot(dataset.ophys_timestamps, corrected_fluorescence[corrected_fluorescence.cell_roi_id==cell_roi_id].corrected_fluorescence.values[0], color='g', label='fluor')
    ax[3].set_ylabel('corrected fluorescence')
    ax[3].legend()

    filename = utils.get_metadata_string(metadata)
    filename = filename+'_'+str(cell_roi_id)
    fig.suptitle(filename, x=0.5, y=1.)
    if save_figure:
        save_dir = loading.get_single_cell_plots_dir()
        utils.save_figure(fig, figsize, save_dir, 'cell_roi_traces_and_masks', filename)