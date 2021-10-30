"""
Created on Thursday September 23 2021

@author: marinag
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.ophys.response_analysis.response_processing as rp
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': True, 'ytick.left': True, })
sns.set_palette('deep')


def plot_population_averages_for_conditions(multi_session_df, df_name, timestamps, axes_column, hue_column, project_code,
                                            use_events=True, filter_events=True, palette=None, data_type='events',
                                            horizontal=True, xlim_seconds=None, save_dir=None, folder=None):
    if palette is None:
        palette = sns.color_palette()

    sdf = multi_session_df.copy()

    # remove traces with incorrect length - why does this happen?
    sdf = sdf.reset_index(drop=True)
    indices = [index for index in sdf.index if len(sdf.iloc[index].mean_trace) == len(sdf.mean_trace.values[0])]
    sdf = sdf.loc[indices]

    if xlim_seconds is None:
        xlim_seconds = [timestamps[0], timestamps[-1]]
    if use_events or filter_events:
        ylabel = 'population response'
    elif 'dFF' in data_type:
        ylabel = 'dF/F'
    elif 'events' in data_type:
        ylabel = 'response'
    elif 'pupil_area' in data_type:
        ylabel = 'pupil area (pix^2)'
    elif 'running' in data_type:
        ylabel = 'running speed (cm/s)'
    if 'omission' in df_name:
        omitted = True
        change = False
        xlabel = 'time after omission (s)'
    elif 'trials' in df_name:
        omitted = False
        change = True
        xlabel = 'time after change (s)'
    else:
        omitted = False
        change = False
        xlabel = 'time (s)'

    hue_conditions = np.sort(sdf[hue_column].unique())
    axes_conditions = np.sort(sdf[axes_column].unique())[::-1]
    if horizontal:
        figsize = (6 * len(axes_conditions), 4)
        fig, ax = plt.subplots(1, len(axes_conditions), figsize=figsize, sharey=False)
    else:
        figsize = (5, 3.5 * len(axes_conditions))
        fig, ax = plt.subplots(len(axes_conditions), 1, figsize=figsize, sharey=False)
    ax = ax.ravel()
    for i, axis in enumerate(axes_conditions):
        for c, hue in enumerate(hue_conditions):
            traces = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)].mean_trace.values
            #             traces = [trace for trace in traces if np.amax(trace) < 4]
            ax[i] = utils.plot_mean_trace(np.asarray(traces), timestamps, ylabel=ylabel,
                                          legend_label=hue, color=palette[c], interval_sec=1,
                                          xlim_seconds=xlim_seconds, ax=ax[i])
            ax[i] = utils.plot_flashes_on_trace(ax[i], timestamps, change=change, omitted=omitted)
            ax[i].axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
            ax[i].set_title(axis)
            ax[i].set_xlim(xlim_seconds)
            ax[i].set_xlabel(xlabel)
            if horizontal:
                ax[i].set_ylabel('')
            else:
                ax[i].set_ylabel(ylabel)
                ax[i].set_xlabel('')
    if horizontal:
        ax[0].set_ylabel(ylabel)
    else:
        ax[i].set_xlabel(xlabel)
    ax[i].legend(loc='upper left', fontsize='x-small')
    if change:
        trace_type = 'change'
    elif omitted:
        trace_type = 'omission'
    else:
        trace_type = 'unknown'
    plt.suptitle('population average ' + trace_type + ' response - ' + project_code[14:], x=0.52, y=1.04, fontsize=18)
    fig.tight_layout()

    if save_dir:
        fig_title = trace_type + '_population_average_response_' + project_code[14:] + '_' + axes_column + '_' + hue_column
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)


def get_timestamps_for_response_df_type(cache, experiment_id, df_name):
    """
    get timestamps from response_df
    """

    dataset = cache.get_behavior_ophys_experiment(experiment_id)
    analysis = ResponseAnalysis(dataset)
    response_df = analysis.get_response_df(df_name=df_name)
    timestamps = response_df.trace_timestamps.values[0]
    print(len(timestamps))

    return timestamps


def plot_cell_response_heatmap(data, timestamps, xlabel='time after change (s)', vmax=0.05,
                               microscope='Multiscope', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.heatmap(data, cmap='magma', linewidths=0, linecolor='white', square=False,
                     vmin=0, vmax=vmax, robust=True, cbar=True,
                     cbar_kws={"drawedges": False, "shrink": 1, "label": 'response'}, ax=ax)
    ax.vlines(x=5 * 11, ymin=0, ymax=len(data), color='w', linestyle='--')

    if microscope == 'Multiscope':
        ax.set_xticks(np.arange(0, 10 * 11, 11))
        ax.set_xticklabels(np.arange(-5, 5, 1))
    ax.set_xlim(3 * 11, 7 * 11)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('cells')
    ax.set_ylim(0, len(data))
    ax.set_yticks(np.arange(0, len(data), 100))
    ax.set_yticklabels(np.arange(0, len(data), 100))

    return ax


def plot_response_heatmaps_for_conditions(multi_session_df, df_name, timestamps,
                                          row_condition, col_condition, use_events, filter_events, project_code,
                                          microscope='Multiscope', vmax=0.05, xlim_seconds=None, match_cells=False,
                                          save_dir=None, folder=None):
    sdf = multi_session_df.copy()

    # remove traces with incorrect length - why does this happen?
    sdf = sdf.reset_index(drop=True)
    indices = [index for index in sdf.index if len(sdf.iloc[index].mean_trace) == len(sdf.mean_trace.values[0])]
    sdf = sdf.loc[indices]

    if 'omission' in df_name:
        xlabel = 'time after omission (s)'
        trace_type = 'omitted'
    elif 'change' in df_name:
        xlabel = 'time after change (s)',
        trace_type = 'change'
    else:
        xlabel = 'time (s)'
        trace_type = 'unknown'

    row_conditions = np.sort(sdf[row_condition].unique())
    col_conditions = np.sort(sdf[col_condition].unique())

    print(len(col_conditions), len(row_conditions))
    figsize = (4 * len(col_conditions), 4 * len(row_conditions))
    fig, ax = plt.subplots(len(row_conditions), len(col_conditions), figsize=figsize, sharex=True)
    ax = ax.ravel()

    i = 0
    for r, row in enumerate(row_conditions):
        row_sdf = sdf[(sdf[row_condition] == row)]
        for c, col in enumerate(col_conditions):

            if row == 'Excitatory':
                interval = 1000
                vmax = 0.01
            elif row == 'Vip Inhibitory':
                interval = 200
                vmax = 0.02
            elif row == 'Sst Inhibitory':
                interval = 100
                vmax = 0.03
            else:
                interval = 200

            tmp = row_sdf[(row_sdf[col_condition] == col)]
            tmp = tmp.reset_index()
            if match_cells:
                if c == 0:
                    tmp = tmp.sort_values(by='mean_response', ascending=True)
                    order = tmp.index.values
                else:
                    tmp = tmp.loc[order]
            else:
                tmp = tmp.sort_values(by='mean_response', ascending=True)
            data = pd.DataFrame(np.vstack(tmp.mean_trace.values), columns=timestamps)

            ax[i] = plot_cell_response_heatmap(data, timestamps, vmax=vmax, xlabel=xlabel,
                                               microscope=microscope, ax=ax[i])
            ax[i].set_title(row + '\n' + col)

            ax[i].set_yticks(np.arange(0, len(data), interval))
            ax[i].set_yticklabels(np.arange(0, len(data), interval))
            ax[i].set_xlim(xlim_seconds)
            if r == len(row_conditions):
                ax[i].set_xlabel(xlabel)
            else:
                ax[i].set_xlabel('')
            i += 1

    # plt.suptitle('response_heatmap '+trace_type+' response - '+project_code[14:], x=0.52, y=1.04, fontsize=18)
    fig.tight_layout()

    if save_dir:
        fig_title = trace_type + '_response_heatmap_' + project_code[14:] + '_' + col_condition + '_' + row_condition
        utils.save_figure(fig, figsize, save_dir, folder, fig_title)


def plot_behavior_timeseries(dataset, start_time, duration_seconds=20, save_dir=None, ax=None):
    """
    Plots licking behavior, rewards, running speed, and pupil area for a defined window of time
    """

    xlim_seconds = [start_time - (duration_seconds / 4.), start_time + duration_seconds * 2]

    lick_timestamps = dataset.licks.timestamps.values
    licks = np.ones(len(lick_timestamps))
    licks[:] = -2

    reward_timestamps = dataset.rewards.timestamps.values
    rewards = np.zeros(len(reward_timestamps))
    rewards[:] = -4

    running_speed = dataset.running_speed.speed.values
    running_timestamps = dataset.running_speed.timestamps.values

    eye_tracking = dataset.eye_tracking.copy()
    pupil_diameter = eye_tracking.pupil_width.values
    pupil_diameter[eye_tracking.likely_blink == True] = np.nan
    pupil_timestamps = eye_tracking.timestamps.values

    if ax is None:
        figsize = (15, 2.5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = sns.color_palette()

    ln0 = ax.plot(lick_timestamps, licks, '|', label='licks', color=colors[3], markersize=10)
    ln1 = ax.plot(reward_timestamps, rewards, 'o', label='rewards', color=colors[9], markersize=10)

    ln2 = ax.plot(running_timestamps, running_speed, label='running_speed', color=colors[2], zorder=100)
    ax.set_ylabel('running speed\n(cm/s)')
    ax.set_ylim(ymin=-8)

    ax2 = ax.twinx()
    ln3 = ax2.plot(pupil_timestamps, pupil_diameter, label='pupil_diameter', color=colors[4], zorder=0)

    ax2.set_ylabel('pupil diameter \n(pixels)')
    #     ax2.set_ylim(0, 200)

    axes_to_label = ln0 + ln1 + ln2 + ln3  # +ln4
    labels = [label.get_label() for label in axes_to_label]
    ax.legend(axes_to_label, labels, bbox_to_anchor=(1, 1), fontsize='small')

    ax = sf.add_stim_color_span(dataset, ax, xlim=xlim_seconds)

    ax.set_xlim(xlim_seconds)
    ax.set_xlabel('time in session (seconds)')
    metadata_string = utils.get_metadata_string(dataset.metadata)
    ax.set_title(metadata_string)

    # ax.tick_params(which='both', bottom=True, top=False, right=False, left=True,
    #                 labelbottom=True, labeltop=False, labelright=True, labelleft=True)
    # ax2.tick_params(which='both', bottom=True, top=False, right=True, left=False,
    #                 labelbottom=True, labeltop=False, labelright=True, labelleft=True)
    if save_dir:
        folder = 'behavior_timeseries'
        utils.save_figure(fig, figsize, save_dir, folder, metadata_string + '_' + str(int(start_time)),
                          formats=['.png'])



def plot_matched_roi_and_trace(ophys_container_id, cell_specimen_id, use_events=False, filter_events=False,
                               save_figure=True):
    """
    Generates plots characterizing single cell activity in response to stimulus, omissions, and changes.
    Compares across all sessions in a container for each cell, including the ROI mask across days.
    Useful to validate cell matching as well as examine changes in activity profiles over days.
    """
    experiments_table = loading.get_platform_paper_experiment_table()
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    container_expts = container_expts.sort_values(by=['experience_level'])
    expts = np.sort(container_expts.index.values)

    if use_events:
        if filter_events:
            suffix = 'filtered_events'
        else:
            suffix = 'events'
    else:
        suffix = 'dff'

    n = len(expts)
    figsize = (9, 6)
    fig, ax = plt.subplots(2, n, figsize=figsize, sharey='row')
    ax = ax.ravel()
    print('ophys_container_id:', ophys_container_id)
    for i, ophys_experiment_id in enumerate(expts):
        print('ophys_experiment_id:', ophys_experiment_id)
        try:
            dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)
            if cell_specimen_id in dataset.dff_traces.index:

                ct = dataset.cell_specimen_table.copy()
                cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id
                roi_masks = dataset.roi_masks.copy()  # save this to use if subsequent session is missing the ROI
                ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                          spacex=30, spacey=30, show_mask=True, ax=ax[i])
                ax[i].set_title(container_expts.loc[ophys_experiment_id].experience_level)

                analysis = ResponseAnalysis(dataset, use_events=use_events, filter_events=filter_events,
                                            use_extended_stimulus_presentations=False)
                sdf = ut.get_mean_df(analysis.get_response_df(df_name='stimulus_response_df'), analysis=analysis,
                                     conditions=['cell_specimen_id', 'is_change'], flashes=True, omitted=False,
                                     get_reliability=False, get_pref_stim=False,
                                     exclude_omitted_from_pref_stim=True)

                window = rp.get_default_stimulus_response_params()["window_around_timepoint_seconds"]
                cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == True)]
                ax[i + n] = sf.plot_mean_trace_from_mean_df(cell_data,
                                                            frame_rate=analysis.ophys_frame_rate, ylabel='response',
                                                            legend_label=None, color='gray', interval_sec=0.5,
                                                            xlims=window, ax=ax[i + n])

                ax[i + n] = sf.plot_flashes_on_trace(ax[i + n], analysis, window=window, trial_type=None,
                                                     omitted=False,
                                                     alpha=0.15, facecolor='gray')
                ax[i + n].set_title('')
            else:
                # plot the max projection image with the xy location of the previous ROI
                # this will fail if the familiar session is the one without the cell matched
                ax[i] = sf.plot_cell_zoom(roi_masks, dataset.max_projection, cell_roi_id,
                                          spacex=30, spacey=30, show_mask=False, ax=ax[i])
                ax[i].set_title(container_expts.loc[ophys_experiment_id].experience_level)

            metadata_string = utils.get_metadata_string(dataset.metadata)

            fig.tight_layout()
            fig.suptitle(str(cell_specimen_id) + '_' + metadata_string, x=0.52, y=1.02,
                         horizontalalignment='center', fontsize=16)
        except Exception as e:
            print('problem for cell_specimen_id:', cell_specimen_id, ', ophys_experiment_id:', ophys_experiment_id)
            print(e)
    if save_figure:
        save_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/cell_matching'
        utils.save_figure(fig, figsize, save_dir, 'mask_mean_response_exp_levels', str(
            cell_specimen_id) + '_' + metadata_string + '_' + suffix)
        plt.close()


# examples
if __name__ == '__main__':

    import visual_behavior.data_access.loading as loading
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

    # load cache
    cache_dir = loading.get_platform_analysis_cache_dir()
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir)
    experiments_table = loading.get_platform_paper_experiment_table()

    # load multi_session_df
    df_name = 'omission_response_df'
    conditions = ['cell_specimen_id']
    use_events = True
    filter_events = True

    multi_session_df = loading.get_multi_session_df(cache_dir, df_name, conditions, experiments_table,
                                                    use_events=use_events, filter_events=filter_events)

    # limit to platform paper dataset
    multi_session_df = multi_session_df[multi_session_df.ophys_experiment_id.isin(experiments_table.index.values)]
    # merge with metadata
    multi_session_df = multi_session_df.merge(experiments_table, on='ophys_experiment_id')

    # set project code & df_name to plot
    project_code = 'VisualBehaviorMultiscope'
    df_name = 'omission_response_df'

    # get timestamps for population average
    experiment_id = experiments_table[experiments_table.project_code == project_code].index.values[9]
    timestamps = get_timestamps_for_response_df_type(cache, experiment_id, df_name)

    # plot population average for experience_level
    axes_column = 'cell_type'
    hue_column = 'experience_level'
    palette = utils.get_experience_level_colors()
    xlim_seconds = [-1.8, 2.25]

    df = multi_session_df[multi_session_df.project_code == project_code]
    plot_population_averages_for_conditions(df, df_name, timestamps,
                                            axes_column, hue_column, palette,
                                            use_events=True, filter_events=True, xlim_seconds=xlim_seconds,
                                            horizontal=True, save_dir=None, folder=None)


