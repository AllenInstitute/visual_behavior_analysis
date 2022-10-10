import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

def get_container_plots_dir():
    return r'/allen/programs/mindscope/workgroups/learning/ophys/qc_plots/container_plots'


def get_session_plots_dir():
    return r'/allen/programs/mindscope/workgroups/learning/ophys/qc_plots/session_plots'


def get_experiment_plots_dir():
    return r'/allen/programs/mindscope/workgroups/learning/ophys/qc_plots/experiment_plots'


def get_single_cell_plots_dir():
    return r'/allen/programs/mindscope/workgroups/learning/ophys/qc_plots/single_cell_plots'
    # return r'\\allen\programs\mindscope\workgroups\learning\ophys\qc_plots\single_cell_plots'


def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    filename = os.path.join(fig_dir, fig_title)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())


def get_colors_for_session_numbers():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    return reds + blues


def get_experience_level_colors():
    """
    get color map corresponding to Familiar, Novel 1 and Novel >1
    Familiar = red
    Novel 1 = blue
    Novel >1 = lighter blue
    """
    import seaborn as sns

    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    purples = sns.color_palette('Purples_r', 6)[:5][::2]

    colors = [reds[0], blues[0], purples[0]]

    return colors


def lighter(color, percent):
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white - color
    return color + vector * percent


def get_cre_line_colors():
    """
    returns colors in order of Slc17a7, Sst, Vip
    """
    colors = [(255 / 255, 152 / 255, 150 / 255),
              (158 / 255, 218 / 255, 229 / 255),
              (197 / 255, 176 / 255, 213 / 255)]
    return colors


def get_stimulus_color_map(as_rgb=False):
    session_number_colors = get_colors_for_session_numbers()
    session_number_colors_GH = get_colors_for_session_numbers_GH()
    black = np.array([0, 0, 0]).astype(np.uint8)

    stimulus_color_map = {
        'gratings_static': (0.25, 0.25, 0.25),
        'gratings_flashed': (0.5, 0.5, 0.5),
        'images_A': session_number_colors[0],
        'images_A_passive': session_number_colors[2],
        'images_A_habituation': session_number_colors[0],
        'images_B': session_number_colors[3],
        'images_B_passive': session_number_colors[5],
        'images_B_habituation': session_number_colors[3],
        'images_G': session_number_colors_GH[0],
        'images_G_passive': session_number_colors_GH[2],
        'images_G_habituation': session_number_colors_GH[0],
        'images_H': session_number_colors_GH[3],
        'images_H_passive': session_number_colors_GH[5],
    }

    if as_rgb:
        for key in list(stimulus_color_map.keys()):
            stimulus_color_map[key] = np.floor(np.array([x for x in list(stimulus_color_map[key])]) * 255).astype(
                np.uint8)

    return stimulus_color_map


def get_stimulus_phase_color_map(as_rgb=False):
    session_number_colors = get_colors_for_session_numbers()
    session_number_colors_GH = get_colors_for_session_numbers_GH()
    white = np.array([1, 1, 1]).astype(np.uint8)

    training_scale = 0.7
    passive_scale = 0.4

    stimulus_phase_color_map = {
        'gratings_static_training': (0.4, 0.4, 0.4),
        'gratings_flashed_training': (0.7, 0.7, 0.7),
        'images_A_training': (session_number_colors[0] + (white-session_number_colors[0]) * training_scale),
        'images_A_habituation_ophys': (session_number_colors[0] + (white-session_number_colors[0]) * training_scale),
        'images_A_ophys': session_number_colors[0],
        'images_A_passive_ophys': (session_number_colors[0] + (white-session_number_colors[0]) * passive_scale),
        'images_B_training': (session_number_colors[3] + (white-session_number_colors[3]) * training_scale),
        'images_B_habituation_ophys': (session_number_colors[3] + (white - session_number_colors[3]) * training_scale),
        'images_B_ophys': session_number_colors[3],
        'images_B_passive_ophys': (session_number_colors[3] + (white-session_number_colors[3]) * passive_scale),
        'images_G_training': (session_number_colors_GH[0] + (white-session_number_colors_GH[0]) * training_scale),
        'images_G_habituation_ophys': (session_number_colors_GH[0] + (white - session_number_colors_GH[0]) * training_scale),
        'images_G_ophys': session_number_colors_GH[0],
        'images_G_passive_ophys': (session_number_colors_GH[0] + (white-session_number_colors_GH[0]) * passive_scale),
        'images_H_ophys': session_number_colors_GH[3],
        'images_H_passive_ophys': (session_number_colors_GH[3] + (white-session_number_colors_GH[3]) * passive_scale),
    }

    if as_rgb:
        for key in list(stimulus_phase_color_map.keys()):
            stimulus_phase_color_map[key] = np.floor(
                np.array([x for x in list(stimulus_phase_color_map[key])]) * 255).astype(np.uint8)

    return stimulus_phase_color_map


def get_session_type_color_map():
    colors = np.floor(np.array([list(x) for x in get_colors_for_session_numbers()]) * 255).astype(np.uint8)
    black = np.array([0, 0, 0]).astype(np.uint8)


    session_type_color_map = {
        'TRAINING_0_gratings_autorewards_15min': black,
        'TRAINING_1_gratings': lighter(black, 0.2),
        'TRAINING_2_gratings_flashed': lighter(black, 0.4),
        'TRAINING_3_images_A_10uL_reward': lighter(black, 0.6),
        'TRAINING_3_images_B_10uL_reward': lighter(black, 0.6),
        'TRAINING_3_images_G_10uL_reward': lighter(black, 0.6),
        'TRAINING_4_images_A_handoff_lapsed': lighter(black, 0.8),
        'TRAINING_4_images_A_handoff_ready': lighter(black, 0.8),
        'TRAINING_4_images_A_training': lighter(black, 0.8),
        'TRAINING_4_images_B_training': lighter(black, 0.8),
        'TRAINING_4_images_G_training': lighter(black, 0.8),
        'TRAINING_5_images_A_epilogue': lighter(black, 0.8),
        'TRAINING_5_images_A_handoff_lapsed': lighter(black, 0.8),
        'TRAINING_5_images_A_handoff_ready': lighter(black, 0.8),
        'TRAINING_5_images_B_epilogue': lighter(black, 0.8),
        'TRAINING_5_images_B_handoff_lapsed': lighter(black, 0.8),
        'TRAINING_5_images_B_handoff_ready': lighter(black, 0.8),
        'TRAINING_5_images_G_epilogue': lighter(black, 0.8),
        'TRAINING_5_images_G_handoff_lapsed': lighter(black, 0.8),
        'TRAINING_5_images_G_handoff_ready': lighter(black, 0.8),


        'OPHYS_0_images_A_habituation': lighter(colors[0, :], 0.8),
        'OPHYS_1_images_A': colors[0, :],
        'OPHYS_2_images_A_passive': colors[1, :],
        'OPHYS_3_images_A': colors[2, :],
        'OPHYS_4_images_B': colors[3, :],
        'OPHYS_5_images_B_passive': colors[4, :],
        'OPHYS_6_images_B': colors[5, :],

        'OPHYS_1_images_B': colors[3, :],
        'OPHYS_2_images_B_passive': colors[4, :],
        'OPHYS_3_images_B': colors[5, :],
        'OPHYS_4_images_A': colors[0, :],
        'OPHYS_5_images_A_passive': colors[1, :],
        'OPHYS_6_images_A': colors[2, :],

        'OPHYS_0_images_G_habituation': lighter(colors[3, :], 0.8),
        'OPHYS_1_images_G': colors[3, :],
        'OPHYS_2_images_G_passive': colors[4, :],
        'OPHYS_3_images_G': colors[5, :],
        'OPHYS_4_images_H': colors[0, :],
        'OPHYS_5_images_H_passive': colors[1, :],
        'OPHYS_6_images_H': colors[2, :],

        'OPHYS_7_receptive_field_mapping': lighter(black, 0.5),
        'None': black,
        None: black,
        np.nan: black,
        'VisCodingTargetedMovieClips': lighter(black, 0.5),
        'full_field_test': lighter(black, 0.2)}

    return session_type_color_map



def get_location_color(location, project_code):
    colors = sns.color_palette()
    if (project_code == 'VisualBehavior') or (project_code == 'VisualBehaviorTask1B'):
        location_colors = {'Slc17a7_VISp_175': colors[9],
                           'Slc17a7_VISp_375': colors[0],
                           'Vip_VISp_175': colors[4],
                           'Sst_VISp_275': colors[2],
                           'Sst_VISp_290': colors[2]}

    elif (project_code == 'VisualBehaviorMultiscope') or (project_code == 'VisualBehaviorMultiscope4areasx2d'):
        location = location.split('_')
        location = location[0] + '_' + location[1]
        location_colors = {'Slc17a7_VISp': sns.color_palette('Blues_r', 5)[0],
                           'Slc17a7_VISl': sns.color_palette('Blues_r', 5)[1],
                           'Slc17a7_VISal': sns.color_palette('Blues_r', 5)[2],
                           'Slc17a7_VISam': sns.color_palette('Blues_r', 5)[3],
                           'Slc17a7_VISpm': sns.color_palette('Blues_r', 5)[4],
                           'Vip_VISp': sns.color_palette('Purples_r', 5)[0],
                           'Vip_VISl': sns.color_palette('Purples_r', 5)[1],
                           'Vip_VISal': sns.color_palette('Purples_r', 5)[2],
                           'Vip_VISam': sns.color_palette('Purples_r', 5)[3],
                           'Vip_VISpm': sns.color_palette('Purples_r', 5)[4],
                           'Sst_VISp': sns.color_palette('Greens_r', 5)[0],
                           'Sst_VISl': sns.color_palette('Greens_r', 5)[1],
                           'Sst_VISal': sns.color_palette('Greens_r', 5)[2],
                           'Sst_VISam': sns.color_palette('Greens_r', 5)[3],
                           'Sst_VISpm': sns.color_palette('Greens_r', 5)[4]}

    return location_colors[location]


# def lighter(color, percent):
#     color = color * 255
#     color = np.array(color)
#     white = np.array([255, 255, 255])
#     return color + (white * percent)
#

def make_color_transparent(rgb_color, background_rgb=[255, 255, 255], alpha=0.5):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb_color, background_rgb)]


def placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1], wspace=None, hspace=None, sharex=False, sharey=False):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec

    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively

    returns:
        subaxes handles
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100, 100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0], dim[1],
                                                  subplot_spec=outer_grid[int(100 * yspan[0]):int(100 * yspan[1]),
                                                                          # flake8: noqa: E999
                                                                          int(100 * xspan[0]):int(100 * xspan[1])], wspace=wspace,
                                                  hspace=hspace)  # flake8: noqa: E999

    # NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0] * [dim[1] * [
        fig]]  # filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            if row > 0 and sharex == True:
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:
                share_y_with = inner_ax[row][0]
            else:
                share_y_with = None

            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx], sharex=share_x_with, sharey=share_y_with)
            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax


def plot_flashes_on_trace(ax, timestamps, change=None, omitted=False, alpha=0.075, facecolor='gray'):
    """
    plot stimulus flash durations on the given axis according to the provided timestamps
    """
    stim_duration = 0.2502
    blank_duration = 0.5004
    change_time = 0
    start_time = timestamps[0]
    end_time = timestamps[-1]
    interval = (blank_duration + stim_duration)
    # after time 0
    if omitted:
        array = np.arange((change_time + interval), end_time, interval) # image array starts at the next interval
        # plot a dashed line where the stimulus time would have been
        ax.axvline(x=change_time, ymin=0, ymax=1, linestyle='--', color=sns.color_palette()[9], linewidth=1.5)
    else:
        array = np.arange(change_time, end_time, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        if change and (i == 0):
            change_color = sns.color_palette()[0]
            ax.axvspan(amin, amax, facecolor=change_color, edgecolor='none', alpha=alpha*1.5, linewidth=0, zorder=1)
        else:
            ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    # if change == True:
    #     alpha = alpha / 2.
    else:
        alpha
    # before time 0
    array = np.arange(change_time, start_time - interval, -interval)
    array = array[1:]
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_mean_trace(traces, timestamps, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlim_seconds=[-2, 2],
                    plot_sem=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        trace = np.mean(traces, axis=0)
        sem = (np.std(traces)) / np.sqrt(float(len(traces)))
        ax.plot(timestamps, trace, label=legend_label, linewidth=2, color=color)
        if plot_sem:
            ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.5, color=color)
        ax.set_xticks(np.arange(int(timestamps[0]), int(timestamps[-1]) + 1, interval_sec))
        ax.set_xlim(xlim_seconds)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_stimulus_response_df_trace(stimulus_response_df, time_window=[-1, 1], change=True, omitted=False,
                                    ylabel=None, legend_label=None, title=None, color='k', ax=None):
    """
    Plot average +/- sem trace for a subset of a stimulus_response_df, loaded via loading.get_stimulus_response_df()
    or directly from mindscope_utilities.visual_behavior_ophys.data_formatting.get_stimulus_response_df()
    :param stimulus_response_df:
    :param time_window:
    :param change:
    :param omitted:
    :param ylabel:
    :param legend_label:
    :param title:
    :param color:
    :param ax:
    :return:
    """
    traces = np.vstack(stimulus_response_df.trace.values)
    timestamps = stimulus_response_df.trace_timestamps.values[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax = plot_mean_trace(traces, timestamps, ylabel=ylabel, legend_label=legend_label, color=color,
                         interval_sec=1, xlim_seconds=time_window, plot_sem=True, ax=ax)

    ax = plot_flashes_on_trace(ax, timestamps, change=change, omitted=omitted, alpha=0.15, facecolor='gray')

    if title:
        ax.set_title(title)

    return ax


def plot_mean_trace_from_mean_df(cell_data, frame_rate=31., ylabel='dF/F', legend_label=None, color='k', interval_sec=1,
                                 xlims=[-4, 4], ax=None, plot_sem=True, width=3):
    """
    plot mean trace for one row in a trial averaged dataframe generated by vba.ophys.response_analysis.utilities.get_mean_df()
    dataframe must have 'mean_trace' and 'sem_trace' as columns
    :param cell_data:
    :param frame_rate:
    :param ylabel:
    :param legend_label:
    :param color:
    :param interval_sec:
    :param xlims:
    :param ax:
    :param plot_sem:
    :param width:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()
    trace = cell_data.mean_trace.values[0]
    timestamps = cell_data.trace_timestamps.values[0]
    sem = cell_data.sem_trace.values[0]
    ax.plot(timestamps, trace, label=legend_label, linewidth=width, color=color)
    if plot_sem:
        ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.5, color=color)
    ax.set_xticks(np.arange(int(timestamps[0]), int(timestamps[-1]) + 1, interval_sec))
    if xlims:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim([timestamps[0], timestamps[-1]])
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax



def get_metadata_string(metadata):
    """
    Create a string of metadata information to be used in filenames and figure titles.
    Includes information such as experiment_id, cre_line, acquisition_date, rig_id, etc
    :param metadata: metadata attribute of dataset object
    :return:
    """
    m = metadata.copy()
    metadata_string = str(m['mouse_id']) + '_' + str(m['ophys_container_id']) + '_' + m['cre_line'].split('-')[0] + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth']) + '_' + m['session_type']
    return metadata_string


def get_container_metadata_string(metadata):
    import visual_behavior.data_access.utilities as utilities
    m = metadata
    # genotype = m['cre_line'].split('-')[0]
    genotype = utilities.get_simple_genotype(m['full_genotype'])
    # add _dox if mouse is a dox mouse
    dox_mice = utilities.get_list_of_dox_mice()
    if str(m['mouse_id']) in dox_mice:
        genotype = genotype+'_dox'
    metadata_string = str(m['mouse_id']) + '_' + str(m['experiment_container_id']) + '_' + genotype + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth'])
    return metadata_string


def get_metadata_for_row_of_multi_session_df(df):
    if len(df) > 1:
        metadata = {}
        metadata['mouse_id'] = df.mouse_id.unique()[0]
        metadata['experiment_container_id'] = df.ophys_container_id.unique()[0]
        metadata['cre_line'] = df.cre_line.unique()[0]
        metadata['targeted_structure'] = df.targeted_structure.unique()[0]
        metadata['imaging_depth'] = df.imaging_depth.unique()[0]
    else:
        metadata = {}
        metadata['mouse_id'] = df.mouse_id
        metadata['experiment_container_id'] = df.ophys_container_id
        metadata['cre_line'] = df.cre_line
        metadata['targeted_structure'] = df.targeted_structure
        metadata['imaging_depth'] = df.imaging_depth
    return metadata


def get_conditions_string(data_type, conditions):
    """
    creates a string containing the data_type and conditions corresponding to a given multi_session_df.
    ignores first element in conditions which is usually 'cell_specimen_id' or 'ophys_experiment_id'
    :param data_type: 'events', 'filtered_events', 'dff'
    :param conditions: list of conditions used to group for averaging in multi_session_df
                        ex: ['cell_specimen_id', 'is_change', 'image_name'], or ['cell_specimen_id', 'engagement_state', 'omitted']
    """

    if len(conditions) == 6:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + \
                            conditions[4] + '_' + conditions[5]
    elif len(conditions) == 5:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + \
                            conditions[4]
    elif len(conditions) == 4:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3]
    elif len(conditions) == 3:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2]
    elif len(conditions) == 2:
        conditions_string = data_type + '_' + conditions[1]
    elif len(conditions) == 1:
        conditions_string = data_typ + '_' + conditions[0]

    return conditions_string