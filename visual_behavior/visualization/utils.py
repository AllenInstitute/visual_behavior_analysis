import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def get_container_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots'

def get_session_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots'

def get_experiment_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/experiment_plots'

def get_single_cell_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/single_cell_plots'


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
    colors = [(255/255,152/255,150/255),
              (158/255, 218/255, 229/255),
              (197/255, 176/255, 213/255)]
    return colors


def get_session_type_color_map():
    colors = np.floor(np.array([list(x) for x in get_colors_for_session_numbers()]) * 255).astype(np.uint8)
    black = np.array([0, 0, 0]).astype(np.uint8)

    session_type_color_map = {
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



def plot_flashes_on_trace(ax, timestamps, change=None, omitted=False, alpha=0.15, facecolor='gray'):
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
        array = np.arange((change_time + interval), end_time, interval)
    else:
        array = np.arange(change_time, end_time, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    if change == True:
        alpha = alpha * 3
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


def plot_mean_trace(traces, timestamps, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlim_seconds=[-2,2],
                    plot_sem=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        trace = np.nanmean(traces, axis=0)
        sem = (np.nanstd(traces)) / np.sqrt(float(len(traces)))
        ax.plot(timestamps, trace, label=legend_label, linewidth=2, color=color)
        if plot_sem:
            ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.5, color=color)
        ax.set_xticks(np.arange(int(timestamps[0]), int(timestamps[-1])+1, interval_sec))
        ax.set_xlim(xlim_seconds)
        ax.set_xlabel('time (sec)')
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
    metadata_string = str(m['mouse_id']) + '_' + str(m['ophys_experiment_id']) + '_' + m['cre_line'].split('-')[0] + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth']) + '_' + m['session_type']
    return metadata_string


def get_container_metadata_string(metadata):
    m = metadata
    metadata_string = str(m['mouse_id'])+'_'+str(m['experiment_container_id'])+'_'+m['cre_line'].split('-')[0]+'_'+m['targeted_structure']+'_'+str(m['imaging_depth'])
    return metadata_string