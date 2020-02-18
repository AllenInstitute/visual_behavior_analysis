import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def gen_colors_for_session_numbers():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    return reds+blues

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
        for t in xtlines+ytlines:
            t.set_visible(False)
    for t in xtlines[1::2]+ytlines[1::2]:
        t.set_visible(False)
    if not yaxis:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ytlines = ax.get_yticklines()
        for t in ytlines:
            t.set_visible(False)