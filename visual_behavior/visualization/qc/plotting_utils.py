import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from visual_behavior.visualization.qc import data_processing as dp


def gen_stage_name_color_palettes():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    greys = sns.color_palette('Greys_r', 6)[:5][::2]
    return reds + blues + greys


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
                              'full_field_test': stage_color_palette[7]}
    return stage_name_colors_dict


def map_stage_name_colors_to_ophys_experiment_ids(dataframe):
    stage_name_colors = gen_ophys_stage_name_colors_dict()
    ophys_experiment_id_stage_name_dict = dp.ophys_experiment_id_stage_name_dict(dataframe)
    experiment_id_color_dict = dict((ophys_experiment_id_stage_name_dict.get(key), value) for (key, value) in stage_name_colors.items())
    return experiment_id_color_dict


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
