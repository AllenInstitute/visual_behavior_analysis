"""
Created on Thursday January 3 2019

@author: marinag
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.visualization.ophys.summary_figures as sf
from visual_behavior.visualization.utils import save_figure
from visual_behavior import utilities as vbut
import seaborn as sns


def plot_histogram(values, label, color='k', range=(0,1), ax=None):
    results, edges = np.histogram(values, normed=True, range=(0,1), bins=50)
    binWidth = edges[1] - edges[0]
    ax.bar(edges[:-1], results*binWidth, binWidth, color=color,label=cre_line, alpha=0.5)
    return ax