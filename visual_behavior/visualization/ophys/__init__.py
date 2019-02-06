"""
Created on Sunday July 15 2018

@author: marinag
"""

import seaborn as sns

def plot_traces_heatmap(traces_df):

    sns.heatmap(traces_df, annot=True)