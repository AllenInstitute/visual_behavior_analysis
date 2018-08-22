"""
Created on Wednesday August 22 2018

@author: marinag
"""
import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# formatting
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_palette('deep')




def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    filename = os.path.join(fig_dir, fig_title)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape')


def plot_traces_heatmap(dff_traces, ax=None):
    if ax is None:
        figsize = (20, 8)
        fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(dff_traces, cmap='magma', vmin=0, vmax=np.percentile(dff_traces[np.isnan(dff_traces)==False], 99))
    ax.set_ylim(0, dff_traces.shape[0])
    ax.set_xlim(0,dff_traces.shape[1])
    ax.set_ylabel('cells')
    ax.set_xlabel('2P frames')
    cb = plt.colorbar(cax, pad = 0.015);
    cb.set_label('dF/F', labelpad=3)
    return ax








