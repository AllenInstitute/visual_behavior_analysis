import pandas as pd
import seaborn as sns

from visual_behavior.data_access import processing as data_processing


def gen_stage_name_color_palettes():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    greys = sns.color_palette('Greys_r', 6)[:5][::2]
    purples = sns.color_palette('RdPu_r', 6)[:5][::2]
    greens = sns.color_palette('Greens_r', 6)[:5][::2]
    return reds + blues + greys + purples + greens


def gen_ophys_stage_name_colors_dict():
    stage_color_palette = gen_stage_name_color_palettes()
    stage_name_colors_dict = {'OPHYS_0_images_A_habituation': stage_color_palette[8],
                              'OPHYS_1_images_A': stage_color_palette[0],
                              'OPHYS_4_images_A': stage_color_palette[0],
                              'OPHYS_2_images_A_passive': stage_color_palette[1],
                              'OPHYS_5_images_A_passive': stage_color_palette[1],
                              'OPHYS_3_images_A': stage_color_palette[2],
                              'OPHYS_6_images_A': stage_color_palette[2],
                              'OPHYS_4_images_B': stage_color_palette[3],
                              'OPHYS_1_images_B': stage_color_palette[3],
                              'OPHYS_5_images_B_passive': stage_color_palette[4],
                              'OPHYS_2_images_B_passive': stage_color_palette[4],
                              'OPHYS_6_images_B': stage_color_palette[5],
                              'OPHYS_3_images_B': stage_color_palette[5],
                              'VisCodingTargetedMovieClips': stage_color_palette[7],
                              'OPHYS_7_receptive_field_mapping': stage_color_palette[7],
                              'full_field_test': stage_color_palette[7],
                              None: stage_color_palette[6],
                              'OPHYS_1_images_G': stage_color_palette[9],
                              'OPHYS_2_images_G_passive': stage_color_palette[10],
                              'OPHYS_3_images_G': stage_color_palette[11],
                              'OPHYS_4_images_H': stage_color_palette[12],
                              'OPHYS_5_images_H_passive': stage_color_palette[13],
                              'OPHYS_6_images_H': stage_color_palette[14]}
    return stage_name_colors_dict


def ophys_experiment_id_stage_name_dict(dataframe):
    """takes a dataframe with the columns "ophys_experiment_id" and "stage_name_lims"
        and returns a dictionary with ophys_experiment_ids as keys and lims stage names
        as values

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        dictionary -- keys: lims stage names (string)
                        values: ophys_experiment_id (9 digit int)
    """

    exp_stage_name_dict = pd.Series(dataframe.ophys_experiment_id.values, index=dataframe.stage_name_lims).to_dict()
    return exp_stage_name_dict


def map_stage_name_colors_to_ophys_experiment_ids(dataframe):
    stage_name_colors = gen_ophys_stage_name_colors_dict()
    ophys_experiment_stage_name_dict = ophys_experiment_id_stage_name_dict(dataframe)
    experiment_id_color_dict = dict((ophys_experiment_stage_name_dict.get(key), value) for (key, value) in stage_name_colors.items())
    return experiment_id_color_dict


def experiment_id_stage_color_dict_for_experiment(ophys_experiment_id):
    experiment_df = data_processing.ophys_experiment_info_df(ophys_experiment_id)
    exp_color_dict = map_stage_name_colors_to_ophys_experiment_ids(experiment_df)
    return exp_color_dict


def experiment_id_stage_color_dict_for_container(ophys_container_id):
    """[summary]

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    container_df = data_processing.ophys_container_info_df(ophys_container_id)
    exp_color_dict = map_stage_name_colors_to_ophys_experiment_ids(container_df)
    return exp_color_dict


def boxoff(ax, keep='left', yaxis=True):
    """
    Hide axis lines, except left and bottom.
    You can specify which axes to keep: 'left' (default), 'right', 'none'.
    """
    ax.spines['top'].set_visible(False)
    xtlines = ax.get_xticklines()
    ytlines = ax.get_yticklines()
    if keep == 'left':
        ax.spines['right'].set_visible(False)
    elif keep == 'right':
        ax.spines['left'].set_visible(False)
    elif keep == 'none':
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        for t in xtlines + ytlines:
            t.set_visible(False)
    for t in xtlines[1::2] + ytlines[1::2]:
        t.set_visible(False)
    if not yaxis:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ytlines = ax.get_yticklines()
        for t in ytlines:
            t.set_visible(False)
