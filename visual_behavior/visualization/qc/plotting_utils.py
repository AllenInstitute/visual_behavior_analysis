import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def gen_stage_name_color_palettes():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    greys = sns.color_palette('Greys_r', 6)[:5][::2]
    return reds + blues + greys


def gen_ophys_stage_name_colors_dict():
    color_palettes = gen_stage_name_color_palettes()
    stage_name_colors_dict = {'OPHYS_1_images_A': color_palettes[0],
                              'OPHYS_4_images_A': color_palettes[0],
                              'OPHYS_2_images_A_passive': color_palettes[1],
                              'OPHYS_5_images_A_passive': color_palettes[1],
                              'OPHYS_3_images_A': color_palettes[2],
                              'OPHYS_6_images_A': color_palettes[2],
                              'OPHYS_4_images_B': color_palettes[3],
                              'OPHYS_1_images_B': color_palettes[3],
                              'OPHYS_5_images_B_passive': color_palettes[4],
                              'OPHYS_2_images_B_passive': color_palettes[4],
                              'OPHYS_6_images_B': color_palettes[5],
                              'OPHYS_3_images_B': color_palettes[5]}
    return stage_name_colors_dict


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
