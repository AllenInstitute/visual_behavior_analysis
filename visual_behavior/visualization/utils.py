import os
import numpy as np
import pandas as pd
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


def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png', '.pdf']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    filename = os.path.join(fig_dir, fig_title)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    for f in formats:
        fig.savefig(filename + f, bbox_inches="tight", transparent=True,
                    orientation='landscape', dpi=300, facecolor=fig.get_facecolor())

def placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1], wspace=None, hspace=None,
                    sharex=False, sharey=False, width_ratios=None, height_ratios=None):
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
                                                  int(100 * xspan[0]):int(100 * xspan[1])],
                                                  wspace=wspace, hspace=hspace,
                                                  width_ratios = width_ratios, height_ratios = height_ratios)  # flake8: noqa: E999

    # NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0] * [dim[1] * [fig]]  # filling the list with figure objects prevents an error when it they are later replaced by axis handles
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

            if row > 0 and sharey == 'col':
                share_y_with = inner_ax[0][col]

            if col > 0 and sharey == 'row':
                share_y_with = inner_ax[row][0]

            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx], sharex=share_x_with, sharey=share_y_with)
            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax

def get_colors_for_session_numbers():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    return blues + reds


def get_colors_for_session_numbers_GH():
    purples = sns.color_palette('Purples_r', 6)[:5][::2]
    greens = sns.color_palette('Greens_r', 6)[:5][::2]
    return purples + greens


def get_old_experience_levels():
    experience_levels = ['Familiar', 'Novel 1', 'Novel >1']
    return experience_levels

def get_experience_levels():
    experience_levels = ['Familiar', 'Novel', 'Novel +']
    return experience_levels

def get_new_experience_levels():
    experience_levels = ['Familiar', 'Novel', 'Novel +']
    return experience_levels

# def get_abbreviated_experience_levels(experience_levels):
#     """
#     converts experience level names (ex: 'Novel >1') into short hand versions (ex: 'N>1')
#     abbreviated names are returned in the same order as provided in experience_levels
#     """
#     exp_level_abbreviations = [exp_level.split(' ')[0][0] if len(exp_level.split(' ')) == 1 else exp_level.split(' ')[0][0] + exp_level.split(' ')[1][:2] for exp_level in experience_levels]
#     # exp_level_abbreviations = ['F', 'N', 'N+']
#     return exp_level_abbreviations


def convert_experience_level(experience_level):
    if experience_level == 'Novel 1':
        new_experience_level = 'Novel'
    elif experience_level == 'Novel >1':
        new_experience_level = 'Novel +'
    elif experience_level == 'Familiar':
        new_experience_level = experience_level
    else: 
        new_experience_level = experience_level 
    return new_experience_level


def get_cre_lines():
    cre_lines = ['Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Vip-IRES-Cre']
    return cre_lines


def get_cell_types():
    cell_types = ['Excitatory', 'Sst Inhibitory', 'Vip Inhibitory']
    return cell_types


def get_cell_type_colors():
    '''
    chosen to approximately match the cell type taxonomy colors for L2/3 E, Sst, Vip
    '''
    c = sns.color_palette('colorblind')
    cell_type_colors = [c[2], c[1], c[4]]
    return cell_type_colors


def convert_cre_line_to_cell_type(cre_line):
    if cre_line == 'Slc17a7-IRES2-Cre':
        cell_type = 'Excitatory'
    elif cre_line == 'Sst-IRES-Cre':
        cell_type = 'Sst Inhibitory'
    elif cre_line == 'Vip-IRES-Cre':
        cell_type = 'Vip Inhibitory'
    return cell_type


def get_abbreviated_cell_type(cre_line):
    """
    returns 3 letter cell type name (i.e. 'Exc', 'Sst', 'Vip')
    :param cre_line:
    :return:
    """
    return convert_cre_line_to_cell_type(cre_line)[:3]


def get_abbreviated_experience_levels():
    """
    converts experience level names (ex: 'Novel +') into short hand versions (ex: 'N+')
    abbreviated names are returned in the same order as provided in experience_levels
    """
    #exp_level_abbreviations = [exp_level.split(' ')[0][0] if len(exp_level.split(' ')) == 1 else exp_level.split(' ')[0][0] + exp_level.split(' ')[1][:2] for exp_level in experience_levels]
    exp_level_abbreviations = ['F', 'N', 'N+']
    return exp_level_abbreviations


def get_experience_level_colors():
    """
    get color values corresponding to Familiar, Novel 1 and Novel >1
    Familiar = blue
    Novel 1 = red
    Novel >1 = purple
    """
    import seaborn as sns

    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    purples = sns.color_palette('Purples_r', 6)[:5][::2]

    # colors = [reds[0], blues[0], purples[0]]
    colors = [blues[0], reds[0],  purples[0]] # changing red to be Novel and blue to be Familiar

    return colors


def get_one_experience_level_colors():
    """
    get color values corresponding to Familiar, Novel 1 and Novel >1
    Familiar = blue
    Novel 1 = red
    Novel >1 = purple
    """
    import seaborn as sns

    reds = sns.color_palette('Reds_r', 4)[:3]
    blues = sns.color_palette('Blues_r', 4)[:3]
    purples = sns.color_palette('Purples_r', 4)[:3]

    # colors = [reds[0], blues[0], purples[0]]
    colors = [blues, reds,  purples] # changing red to be Novel and blue to be Familiar

    return colors


def get_experience_level_cmap():
    """
    get color map corresponding to Familiar, Novel 1 and Novel >1
    Familiar = blue
    Novel 1 = red
    Novel >1 = purple
    """

    colors = ['Blues', 'Reds',  'Purples'] 

    return colors


def color_xaxis_labels_by_experience(ax):
    """
    iterates through x-axis tick labels and sets them to experience level colors in an alternating way,
    assuming that the labels are in [F, N, N+]
    """
    c_vals = get_experience_level_colors()
    [t.set_color(i) for (i,t) in zip([c_vals[0], c_vals[1], c_vals[2]], ax.xaxis.get_ticklabels())]


def color_yaxis_labels_by_experience(ax):
    """
    iterates through x-axis tick labels and sets them to experience level colors in an alternating way,
    assuming that the labels are in [F, N, N+]
    """
    c_vals = get_experience_level_colors()
    [t.set_color(i) for (i,t) in zip([c_vals[0], c_vals[1], c_vals[2]], ax.yaxis.get_ticklabels())]


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


def get_cre_line_color_dict():
    """
    returns colors as a dict with cre line as key and color as value
    """
    colors = {'Slc17a7-IRES2-Cre': (255 / 255, 152 / 255, 150 / 255),
              'Sst-IRES-Cre': (158 / 255, 218 / 255, 229 / 255),
              'Vip-IRES-Cre': (197 / 255, 176 / 255, 213 / 255)}
    return colors


def get_stimulus_color_map(as_rgb=False):
    session_number_colors = get_colors_for_session_numbers()
    session_number_colors_GH = get_colors_for_session_numbers_GH()
    black = np.array([0, 0, 0]).astype(np.uint8)

    stimulus_color_map = {
        'gratings': (0.5, 0.5, 0.5),
        'images': session_number_colors[0],
        # 'gratings_static': (0.5, 0.5, 0.5),
        # 'gratings_flashed': (0.25, 0.25, 0.25),
        'gratings_training': (0.5, 0.5, 0.5),
        'gratings_static': (0.6, 0.6, 0.6),
        'gratings_flashed': (0.4, 0.4, 0.4),
        'familiar': session_number_colors[0],
        'novel': session_number_colors[3],
        'familiar_images': session_number_colors[0],
        'novel_images': session_number_colors[3],
        'images_A': session_number_colors[0],
        'images_A_ophys': session_number_colors[0],
        'images_A_passive': session_number_colors[2],
        'images_A_training': sns.color_palette('Reds_r', 6)[:5][::2][1],
        'images_A_habituation': session_number_colors[0],
        'images_B': session_number_colors[3],
        'images_B_ophys': session_number_colors[3],
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
        'images_A_training': (session_number_colors[0] + (white - session_number_colors[0]) * training_scale),
        'images_A_habituation_ophys': (session_number_colors[0] + (white - session_number_colors[0]) * training_scale),
        'images_A_ophys': session_number_colors[0],
        'images_A_passive_ophys': (session_number_colors[0] + (white - session_number_colors[0]) * passive_scale),
        'images_B_training': (session_number_colors[3] + (white - session_number_colors[3]) * training_scale),
        'images_B_habituation_ophys': (session_number_colors[3] + (white - session_number_colors[3]) * training_scale),
        'images_B_ophys': session_number_colors[3],
        'images_B_passive_ophys': (session_number_colors[3] + (white - session_number_colors[3]) * passive_scale),
        'images_G_training': (session_number_colors_GH[0] + (white - session_number_colors_GH[0]) * training_scale),
        'images_G_habituation_ophys': (session_number_colors_GH[0] + (white - session_number_colors_GH[0]) * training_scale),
        'images_G_ophys': session_number_colors_GH[0],
        'images_G_passive_ophys': (session_number_colors_GH[0] + (white - session_number_colors_GH[0]) * passive_scale),
        'images_H_ophys': session_number_colors_GH[3],
        'images_H_passive_ophys': (session_number_colors_GH[3] + (white - session_number_colors_GH[3]) * passive_scale),
    }

    if as_rgb:
        for key in list(stimulus_phase_color_map.keys()):
            stimulus_phase_color_map[key] = np.floor(
                np.array([x for x in list(stimulus_phase_color_map[key])]) * 255).astype(np.uint8)

    return stimulus_phase_color_map

def get_stimulus_phase_color_map(as_rgb=False):
    session_number_colors = get_colors_for_session_numbers()
    session_number_colors_GH = get_colors_for_session_numbers_GH()
    white = np.array([1, 1, 1]).astype(np.uint8)

    training_scale = 0.7
    passive_scale = 0.4

    stimulus_phase_color_map = {
        'gratings_static_training': (0.4, 0.4, 0.4),
        'gratings_flashed_training': (0.7, 0.7, 0.7),
        'images_A_training': (session_number_colors[0] + (white - session_number_colors[0]) * training_scale),
        'images_A_habituation_ophys': (session_number_colors[0] + (white - session_number_colors[0]) * training_scale),
        'images_A_ophys': session_number_colors[0],
        'images_A_passive_ophys': (session_number_colors[0] + (white - session_number_colors[0]) * passive_scale),
        'images_B_training': (session_number_colors[3] + (white - session_number_colors[3]) * training_scale),
        'images_B_habituation_ophys': (session_number_colors[3] + (white - session_number_colors[3]) * training_scale),
        'images_B_ophys': session_number_colors[3],
        'images_B_passive_ophys': (session_number_colors[3] + (white - session_number_colors[3]) * passive_scale),
        'images_G_training': (session_number_colors_GH[0] + (white - session_number_colors_GH[0]) * training_scale),
        'images_G_habituation_ophys': (session_number_colors_GH[0] + (white - session_number_colors_GH[0]) * training_scale),
        'images_G_ophys': session_number_colors_GH[0],
        'images_G_passive_ophys': (session_number_colors_GH[0] + (white - session_number_colors_GH[0]) * passive_scale),
        'images_H_ophys': session_number_colors_GH[3],
        'images_H_passive_ophys': (session_number_colors_GH[3] + (white - session_number_colors_GH[3]) * passive_scale),
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

        'OPHYS_0_images_B_habituation': lighter(colors[0, :], 0.8),
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


def get_behavior_stage_color_map(as_rgb=False):
    """
    create colormap, as rgb or [0,1], corresponding to behavior stages defined in add_behavior_stage_to_behavior_sessions
    (ex: ['gratings_static_training', 'gratings_flashed_training', 'familiar_images_training', )

    """
    session_number_colors = get_experience_level_colors()
    white = np.array([1, 1, 1]).astype(np.uint8)

    training_scale = 0.7
    passive_scale = 0.4

    behavior_stage_color_map = {
        'gratings_static_training': (0.6, 0.6, 0.6),
        'gratings_flashed_training': (0.4, 0.4, 0.4),
        'familiar_images_training': (session_number_colors[0] + (white - session_number_colors[0]) * training_scale),
        'familiar_images_ophys': session_number_colors[0],
        'familiar_images_ophys_passive': (
        session_number_colors[0] + (white - session_number_colors[0]) * passive_scale),
        'novel_images_ophys': session_number_colors[1],
        'novel_images_ophys_passive': (session_number_colors[2] + (white - session_number_colors[2]) * passive_scale),
        'novel +_images_ophys': session_number_colors[2],
        'novel +_images_ophys_passive': (session_number_colors[2] + (white - session_number_colors[2]) * passive_scale),
    }

    if as_rgb:
        for key in list(behavior_stage_color_map.keys()):
            behavior_stage_color_map[key] = np.floor(
                np.array([x for x in list(behavior_stage_color_map[key])]) * 255).astype(np.uint8)

    return behavior_stage_color_map


def get_ophys_stage_color_map(as_rgb=False):
    # session_number_colors = get_colors_for_session_numbers()
    experience_level_colors = get_experience_level_colors()
    gh_colors = get_colors_for_session_numbers_GH()
    white = np.array([1, 1, 1]).astype(np.uint8)

    passive_scale = 0.6
    active_not_in_dataset_scale = 0.4

    ophys_stage_color_map = {
        'familiar_images_in_dataset': experience_level_colors[0],
        'familiar_images': (
                    experience_level_colors[0] + (white - experience_level_colors[0]) * active_not_in_dataset_scale),
        'familiar_images_passive': (experience_level_colors[0] + (white - experience_level_colors[0]) * passive_scale),
        'novel_images_first_novel_in_dataset': experience_level_colors[1],
        'novel +_images_in_dataset': experience_level_colors[2],
        'novel +_images': (experience_level_colors[2] + (white - experience_level_colors[2]) * active_not_in_dataset_scale),
        'novel +_images_passive': (experience_level_colors[2] + (white - experience_level_colors[2]) * passive_scale),
    }

    if as_rgb:
        for key in list(ophys_stage_color_map.keys()):
            ophys_stage_color_map[key] = np.floor(
                np.array([x for x in list(ophys_stage_color_map[key])]) * 255).astype(np.uint8)

    return ophys_stage_color_map


def make_color_transparent(rgb_color, background_rgb=[255, 255, 255], alpha=0.5):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb_color, background_rgb)]


def addSpan(ax, amin, amax, color='k', alpha=0.3, axtype='x'):
    """
    adds a vertical span to an axis
    """
    if axtype == 'x':
            ax.axvspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0)
    if axtype == 'y':
        ax.axhspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0)


def add_stim_color_span(dataset, ax, xlim=None, color=None, label_changes=True, label_omissions=True,
                        annotate_changes=False, max_alpha=0.5):
    """
    adds a vertical span for all stimulus presentations contained within xlim
    xlim is a time in seconds during a behavior session
    if label_changes is True, changes will be blue and all other flashes will be gray
    if label_changes is False, each flash will be colored according to image identity
    if label_omissions is True, a dotted line will be shown at the time of omission
    if a color is provided, all stimulus presentations will be that color
    """
    # set default alpha. If label_changes=True, alphas will be reset below.
    alpha = 0.3
    # get stim table
    stim_table = dataset.stimulus_presentations.copy()
    import visual_behavior.data_access.loading as loading
    stim_table = loading.limit_stimulus_presentations_to_change_detection(stim_table)
    # remove omissions because they dont get labeled
    #     stim_table = stim_table[stim_table.omitted == False].copy()
    # get all images & assign colors (image colors wont be used if a color is provided or if label_changes is True)
    images = np.sort(stim_table[stim_table.omitted == False].image_name.unique())
    image_colors = sns.color_palette("hls", len(images))
    # limit to time window if provided
    if xlim is not None:
        stim_table = stim_table[(stim_table.start_time >= xlim[0]) & (stim_table.end_time <= xlim[1])]
    # loop through stimulus presentations and add a span with appropriate color
    for idx in stim_table.index:
        start_time = stim_table.loc[idx]['start_time']
        end_time = stim_table.loc[idx]['end_time']
        image_name = stim_table.loc[idx]['image_name']
        image_index = stim_table.loc[idx]['image_index']
        if image_name == 'omitted':
            if label_omissions:
                ax.axvline(x=start_time, ymin=0, ymax=1, linestyle='--', color=sns.color_palette()[9])
        else:
            if label_changes:
                if stim_table.loc[idx]['is_change']:  # if its a change, make it blue with higher alpha
                    image_color = sns.color_palette()[0]
                    alpha = max_alpha
                    if annotate_changes:
                        ymin, ymax = ax.get_ylim()
                        ax.annotate(stim_table.loc[idx]['image_name'], xy=(start_time, ymax*2), xycoords='data',
                                    fontsize=6,  va='top', clip_on=False, annotation_clip=False)
                else:  # if its a non-change make it gray with low alpha
                    image_color = 'gray'
                    alpha = max_alpha/2.
            else:
                if color is None:
                    image_color = image_colors[image_index]
                else:
                    image_color = color
            addSpan(ax, start_time, end_time, color=image_color, alpha=alpha)
    return ax

def plot_flashes_on_trace(ax, timestamps, change=None, omitted=False, alpha=0.075, facecolor='gray', linewidth=1.5):
    """
    plot stimulus flash durations on the given axis according to the provided timestamps
    will show repeated images in gray, image changes in blue, and image omissions with a dotted line
    """
    stim_duration = 0.2502
    blank_duration = 0.5004
    change_time = 0
    start_time = timestamps[0]
    end_time = timestamps[-1]
    interval = (blank_duration + stim_duration)
    # after time 0
    if omitted:
        array = np.arange((change_time + interval), end_time, interval)  # image array starts at the next interval
        # plot a dashed line where the stimulus time would have been
        ax.axvline(x=change_time, ymin=0, ymax=1, linestyle='--', color=sns.color_palette()[9], linewidth=linewidth)
    else:
        array = np.arange(change_time, end_time, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        if change and (i == 0):
            change_color = sns.color_palette()[0]
            ax.axvspan(amin, amax, facecolor=change_color, edgecolor='none', alpha=alpha * 1.5, linewidth=0, zorder=1)
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


def plot_mean_trace(traces, timestamps, ylabel='dF/F', legend_label=None, color='k',
                    interval_sec=1, xlim_seconds=[-2, 2], linewidth=1, plot_sem=True, ax=None):
    '''
    compute average and SEM of traces array and plot it on the specified axis
    produces x-axis timestamps based on the xlim_seconds range provided, incremented by the value of interval_sec
    '''
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        trace = np.mean(traces, axis=0)
        sem = (np.std(traces)) / np.sqrt(float(len(traces)))
        ax.plot(timestamps, trace, label=legend_label, linewidth=linewidth, color=color)
        if plot_sem:
            ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.4, color=color)
        ax.set_xticks(np.arange(int(timestamps[0]), int(timestamps[-1]) + 1, interval_sec))
        ax.set_xlim(xlim_seconds)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_mean_trace_from_mean_df(cell_data, ylabel='dF/F', xlabel='time (s)', legend_label=None, color='k', interval_sec=1,
                                 xlims=[-4, 4],  ax=None, plot_sem=True, linewidth=3):

    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    trace = cell_data.mean_trace.values[0]
    timestamps = cell_data.trace_timestamps.values[0]
    sem = cell_data.sem_trace.values[0]
    ax.plot(timestamps, trace, label=legend_label, linewidth=linewidth, color=color)
    if plot_sem:
        ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.5, color=color)
    ax.set_xticks(np.arange(int(timestamps[0]), int(timestamps[-1]) + 1, interval_sec))
    ax.set_xlim(xlims)
    ax.set_xlabel(xlabel)
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


def plot_population_averages_for_conditions(multi_session_df, data_type, event_type, axes_column, hue_column,
                                            project_code=None, timestamps=None, palette=None, sharey=False,
                                            title=None, suptitle=None, xlabel='Time (s)', ylabel='Response',
                                            horizontal=True, xlim_seconds=None, interval_sec=1, legend=True,
                                            save_dir=None, folder=None, suffix='', ax=None):
    '''
    Function to plot a population average response across multiple conditions from a dataframe containing event aligned timeseries,
    where axes_column defines the axes conditions and hue_column defines the colors of traces within each axes condition.
    axes_column and hue_column must be columns of the multi_session_df.
    multi_session_df must contain a column for 'mean_trace' and rows should be individual cells' average responses to a specific condition.
    also works for behavior timeseries, in which case rows are averages across an experiment or subset of an experiment rather than individual cells.

    event_type is one of ['changes', 'omissions', 'images']
    this determines how stimuli will be plotted overlaid with the trace - changes in blue, omissions with dotted line, repeated images in gray

    data_type is one of ['dff', 'events', 'filtered_events', 'running_speed', 'pupil_width', 'lick_rate']

    interval_sec determines the interval of the xtick labels (ex: ticks every 1 second or 0.5 seconds)
    xlim_seconds is the range of x-axis, which must be the same or shorter than the range of the data in the 'mean_response' column of the multi_session_df.
    timestamps can be provided, or inferred from the 'trace_timestamps' column of the multi_session_df.

    event aligned timeseries can be computed using brain_observatory_utilities function 'get_stimulus_response_df' here:
    https://github.com/AllenInstitute/brain_observatory_utilities/blob/main/brain_observatory_utilities/datasets/optical_physiology/data_formatting.py#L441
    Followed by a groupby and mean on the conditions of interest.

    '''

    if palette is None:
        palette = get_experience_level_colors()

    sdf = multi_session_df.copy()

    # get timestamps
    if 'trace_timestamps' in sdf.keys():
        timestamps = sdf.trace_timestamps.values[0]
    elif timestamps is not None:
        timestamps = timestamps
    else:
        print('provide timestamps or provide a multi_session_df with a trace_timestamps column')

    # set formatting options
    if xlim_seconds is None:
        xlim_seconds = [timestamps[0], timestamps[-1]]
    if event_type == 'omissions':
        omitted = True
        change = False
    elif event_type == 'changes':
        omitted = False
        change = True
    else:
        omitted = False
        change = False

    # get conditions to plot
    hue_conditions = np.sort(sdf[hue_column].unique())
    axes_conditions = np.sort(sdf[axes_column].unique())

    # if there is only one axis condition, set n conditions for plotting to 2 so it can still iterate
    if len(axes_conditions) == 1:
        n_axes_conditions = 2
        ax_to_xlabel = 1
    else:
        n_axes_conditions = len(axes_conditions)

    # set plot size depending on what type of data it is
    if data_type in ['dff', 'events', 'filtered_events']:
        if horizontal:
            figsize = (4 * n_axes_conditions, 2.5)
        else:
            figsize = (3, 3 * n_axes_conditions)  # for changes and omissions
    elif data_type in ['running_speed', 'pupil_width', 'lick_rate']:
        if horizontal:
            figsize = (6 * n_axes_conditions, 3)  # for behavior timeseries
        else:
            figsize = (2.5, 3 * n_axes_conditions)  # for image response

    # create axes
    if ax is None:
        format_fig = True
        if horizontal:
            suffix = suffix+'_horiz'
            fig, ax = plt.subplots(1, n_axes_conditions, figsize=figsize, sharey=sharey)
        else:
            fig, ax = plt.subplots(n_axes_conditions, 1, figsize=figsize, sharex=sharey)
    else:
        format_fig = False

    # loop over conditions and plot
    for i, axis in enumerate(axes_conditions):
        for c, hue in enumerate(hue_conditions):
            # try:
            cdf = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)]
            traces = cdf.mean_trace.values
            # plot average of all traces for this condition
            ax[i] = plot_mean_trace(np.asarray(traces), timestamps, ylabel=ylabel,
                                          legend_label=hue, color=palette[c], interval_sec=interval_sec,
                                          xlim_seconds=xlim_seconds, ax=ax[i])
            # plot stimulus timing overlaid on trace
            ax[i] = plot_flashes_on_trace(ax[i], timestamps, change=change, omitted=omitted)

            # color title by experience level if axes are experience levels
            if axes_column == 'experience_level':
                title_colors = get_experience_level_colors()
                ax[i].set_title(axis, color=title_colors[i], fontsize=20)
            else:
                ax[i].set_title(axis)
            if title:
                ax[i].set_title(title)
            ax[i].set_xlim(xlim_seconds)
            ax[i].set_xlabel(xlabel, fontsize=16)
            ax[i].set_ylabel('')
            ax[i].set_xlabel('')
            ax[i].tick_params(axis='both', which='major', labelsize=14)
    # formatting
    if format_fig:
        if horizontal:
            ax[0].set_ylabel(ylabel)
            if n_axes_conditions == 3:
                ax[1].set_xlabel(xlabel)
            else:
                ax[0].set_xlabel(xlabel)
        else:
            ax[1].set_ylabel(ylabel)
            ax[i].set_xlabel(xlabel)
    if legend:
        ax[i].legend(loc='upper center', fontsize='x-small', bbox_to_anchor=(1.5,1))
    if project_code:
        if suptitle is None:
            suptitle = 'population average - ' + data_type + ' response - ' + project_code[14:]
    if suptitle:
        if horizontal:
            y = 1.1
        else:
            y = 0.97
        plt.suptitle(suptitle, x=0.51, y=y, fontsize=18)

    plt.rcParams["savefig.bbox"] = "tight"
    if save_dir:
        plt.rcParams["savefig.bbox"] = "tight"
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        fig_title = 'population_average_' + axes_column + '_' + hue_column + suffix
        save_figure(fig, figsize, save_dir, folder, fig_title)

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
    m = metadata.copy()
    metadata_string = str(m['ophys_container_id']) + '_' + str(m['ophys_experiment_id']) + '_' + m['cre_line'].split('-')[0] + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth'])
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
        conditions_string = data_type + '_' + conditions[0]

    return conditions_string


def get_start_end_time_for_period_with_omissions_and_change(stimulus_presentations, n_flashes=16):
    st = stimulus_presentations.copy()
    indices = st[st.omitted].index.values[20:] # start from the 10th omission
    # get all start times for periods with an omission and change
    start_times = []
    for idx in indices: # loop through omission times
        subset = st.loc[idx:idx+n_flashes-4] #from 4 flashes before omission to 2 flashes before end of window
        if subset.is_change.any():
            start_time = st.loc[idx - 4].start_time # start time is -4 flashes before the omission
            start_times.append(start_time)
    print(len(start_times))
    # pick the 10th time
    start_time = start_times[10]
    end_time = start_time+(0.75*n_flashes)
    return [start_time, end_time]

def get_start_times_for_period_with_hit_and_omission(stimulus_presentations, trials, n_flashes=16):
    st = stimulus_presentations.copy()
    hit_times = trials[trials.hit].change_time.values
    indices = st[st.start_time.isin(hit_times)].index.values # indices of hit times
    # get all start times for periods with hit and check if there is also an omission within n_flashes-4
    start_times = []
    for idx in indices: # loop through omission times
        subset = st.loc[idx:idx+n_flashes-2] #from 4 flashes before omission to 2 flashes before end of window
        if subset.omitted.any():
            start_time = st.loc[idx - 4].start_time # start time is -4 flashes before the change
            start_times.append(start_time)
    print(len(start_times))
    return start_times

def get_start_end_time_for_period_with_event(stimulus_presentations, event_type='is_change',
                                             n_before=4, n_after=8, event_ind_to_choose=10,
                                             two_in_window=False):
    '''

    Get start and end times around a given event based on the requested number of image flashes before and after the event
    if 'two_in_window' is True, function will check whether there are two events within the window,
    if there are, the start and end times will be returned, if there are not two events in the window,
    the returned start and end times will be None

    Parameters
    ----------
    stimulus_presentations: SDK stimulus presentations table
    event_type: binary column in stimulus_presentations that can be used to filter to find relevant timepoints
    n_before: number of flashes before the event to take window for start time
    n_after: number of flashes after the event to take window for dnd time
    event_ind_to_choose: index of event to use (i.e. the 10th change or 15th omission)
    Returns
    -------

    '''

    st = stimulus_presentations.copy()
    index = st[st[event_type] == True].index.values[event_ind_to_choose]
    event_time = st.loc[index].start_time
    start_time = st.loc[index - n_before].start_time
    start_time = np.round(start_time, 1)
    end_time = start_time + (0.75 * (n_after))
    if two_in_window:
        # find time of next change
        ind = st[st['is_change'] == True].index.values[event_ind_to_choose + 1]
        next_start = st.loc[ind].start_time
        # check if next event is in the window defined by n_after flashes
        if (next_start+(0.75*4) - event_time) < (end_time - event_time):
            print('found two changes, event_index:', event_ind_to_choose)
            window = [start_time, end_time]
        else:
            window = None
    else:
        window = [start_time, end_time]
    return window


def get_consecutive_trials_with_omission_in_middle(dataset, flashes_between_changes=16):
    ''' Loops through trials and finds those where a hit and a miss are within a given distance of each other
    and there is an omission between them
    '''

    st = dataset.stimulus_presentations.copy()
    trials = dataset.trials.copy()

    # get all trials where the next one is less than the designated number of flashes between changes
    time_between_changes = flashes_between_changes*0.75
    diffs = np.where(trials.change_time.diff()<time_between_changes)[0] -1

    trials_to_keep = []
    start_times = []
    for diff in diffs: # for all trials that have another change within the selected window
        # if the current change is a hit and the next change is a miss (or vice versa)
        if (trials.loc[diff].hit and trials.loc[diff+1].miss) | (trials.loc[diff].miss and trials.loc[diff+1].hit) :
            # get the start and end times of the two trials
            start_time = trials.loc[diff].change_time
            end_time = trials.loc[diff+1].change_time
            window_stim = st[(st.start_time>start_time) & (st.end_time<end_time)]
            # check whether there is an omission in that window
            if len(window_stim[window_stim.omitted]) > 0:
                # if so, keep it
                trials_to_keep.append(diff)
                start_times.append(start_time-(3*0.75)) # start time to keep is 2 flashes before the first change
    # return trials_to_keep
    return start_times


def get_experiments_matched_across_project_codes(df):
    '''
    To compare response properties across project codes, you need to limit to the areas & depths that are matched
    between Scientifica (which only imaged in VISp at specific depths per cre line) and Multiscope (which imaged
    across VISp and VISl at multiple depths per cre line)

    df = any dataframe containing metadata about area, depth, and cell type, that includes ophys_experiment_id as a column
    '''
    df = df[df.targeted_structure=='VISp']
    exc_oeids_upper = list(df[(df.cell_type=='Excitatory') & (df.binned_depth==175)].ophys_experiment_id.unique())
    exc_oeids_lower = list(df[(df.cell_type=='Excitatory') & (df.binned_depth==375)].ophys_experiment_id.unique())
    sst_oeids = list(df[(df.cell_type=='Sst Inhibitory') & (df.binned_depth==275)].ophys_experiment_id.unique())
    vip_oeids = list(df[(df.cell_type=='Vip Inhibitory') & (df.binned_depth==175)].ophys_experiment_id.unique())

    matched_oeids = exc_oeids_upper + exc_oeids_lower + sst_oeids + vip_oeids
    return matched_oeids

def save_reliable_matched_cells_and_containers_list(matched_cells_table, multi_session_df, event_type='changes'):
    '''
    saves a table of cell_specimen_ids and ophys_container_ids for cells that are
    reliable and responsive over a defined threshold and matched across sessions

    matched_cells_table: ophys_cells_table instance limited to platform paper experiments
                    and cells matched in all 3 sessions
    multi_session_df: dataframe containing trial averaged responses and response metrics for all cells
                    can be aligned to changes, images or omissions
    event_type: type of stimulus event that traces are aligned to and responsiveness is computed for
                    can be one of ['changes', 'images', 'omissions]
    '''


    n_reliable_cells_threshold = 4

    all_matched_reliable_cells = pd.DataFrame()
    for cell_type in get_cell_types():
        # set thresholds
        if cell_type == 'Excitatory':
            fraction_sig = 0.1
            mean_response = 0.05
        elif cell_type == 'Sst Inhibitory':
            fraction_sig = 0.1
            mean_response = 0.01
        elif cell_type == 'Vip Inhibitory':
            fraction_sig = 0.1
            mean_response = 0.05

        # get cells with reliability and mean response over threshold
        reliable_cells = multi_session_df[(multi_session_df.fraction_significant_p_value_gray_screen>fraction_sig) &
                (multi_session_df.mean_response > mean_response) &
                (multi_session_df.cell_type==cell_type)]
        # make sure they are matched across sessions
        matched_reliable_cells = reliable_cells[reliable_cells.cell_specimen_id.isin(matched_cells_table.cell_specimen_id.unique())]
        # make sure we only include containers with at least some # of reliable cells
        containers = matched_reliable_cells.groupby('ophys_container_id').count().rename(columns={'cell_specimen_id': 'n_cells'})
        ophys_container_ids = containers[containers.n_cells>=n_reliable_cells_threshold].index.values
        matched_reliable_cells = matched_reliable_cells[matched_reliable_cells.ophys_container_id.isin(ophys_container_ids)]
        # add the threshold values to table
        matched_reliable_cells['fraction_significant_threshold'] = fraction_sig
        matched_reliable_cells['mean_response_threshold'] = mean_response
        # print out how many there are
        print(cell_type)
        print(len(matched_reliable_cells.ophys_container_id.unique()))
        print(len(matched_reliable_cells.cell_specimen_id.unique()))
        # aggregate
        all_matched_reliable_cells = pd.concat([all_matched_reliable_cells, matched_reliable_cells])

    # limit to relevant columns and save
    all_matched_reliable_cells = all_matched_reliable_cells[['cell_specimen_id', 'ophys_container_id', 'ophys_experiment_id',
                                                            'fraction_significant_threshold', 'mean_response_threshold']]
    save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_cache_new'
    filename = 'reliable_matched_cells_and_containers_'+event_type+'.csv'
    all_matched_reliable_cells.to_csv(os.path.join(save_dir, filename))

    return all_matched_reliable_cells