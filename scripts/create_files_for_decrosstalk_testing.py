#!/usr/bin/env python

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

import visual_behavior.data_access.utilities as utilities
import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.visualization.utils as utils

if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]

### generate files
    # dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=True)
    # analysis = ResponseAnalysis(dataset, overwrite_analysis_files=True)
    # # save full set of traces
    # dff_traces = dataset.dff_traces.copy()
    # # dff_traces.to_hdf(os.path.join(analysis.analysis_dir, 'dff_traces_post_decrosstalk.h5'), key='df') #ROI ids mixed up
    # dff_traces.to_hdf(os.path.join(analysis.analysis_dir, 'dff_traces_with_decrosstalk.h5'), key='df') #ROI ids should be fixed
    # # identify NaN traces before response_df creation
    # dff_traces = dataset.dff_traces.copy()
    # non_nan_inds = [row for row in range(len(dff_traces)) if np.isnan(dff_traces.iloc[row].dff[0]) == False]
    # # dataset.dff_traces = dff_traces.iloc[non_nan_inds]
    # valid_cell_specimen_ids = dff_traces.iloc[non_nan_inds].index.values
    # # get stimulus locked responses & overwrite in cache_dir to compare to non-decrosstalked examples
    # sdf = analysis.get_response_df(df_name='stimulus_response_df')
    # sdf = sdf[sdf.cell_specimen_id.isin(valid_cell_specimen_ids)]

### plot output comparison

# cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'

    candidates = [file for file in os.listdir(cache_dir) if str(experiment_id) in file]
    if len(candidates) == 1:
        analysis_folder = candidates[0]
        analysis_dir = os.path.join(cache_dir, analysis_folder)
    else:
        print('no analysis folder for', experiment_id)

    # These files contain the output of SDK session object attribute dff_traces,
    # saved separately for production output currently in lims (without decrosstalk) and Wayne's dev lims environment (with decrosstalk)
    dff_traces_pre_decrosstalk = pd.read_hdf(os.path.join(analysis_dir, 'dff_traces_without_decrosstalk.h5'), key='df')
    dff_traces_post_decrosstalk = pd.read_hdf(os.path.join(analysis_dir, 'dff_traces_with_decrosstalk.h5'), key='df')

    # get metadata for plot title
    dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=True)  # set to False to limit to valid ROIs (post filtering)

    for cell_roi_id in dff_traces_pre_decrosstalk.cell_roi_id.values:

        figsize = (20, 15)
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax[0].plot(dff_traces_post_decrosstalk[dff_traces_post_decrosstalk.cell_roi_id == cell_roi_id].dff.values[0],
                   color='blue', label='dev - with decrosstalk')
        ax[0].plot(dff_traces_pre_decrosstalk[dff_traces_pre_decrosstalk.cell_roi_id == cell_roi_id].dff.values[0],
                   color='black', label='prod - without decrosstalk')
        ax[0].set_title('cell_roi_id: ' + str(cell_roi_id) + ', dF/F traces before and after decrosstalk')
        ax[0].legend(loc='upper right')

        ax[1].plot(dff_traces_post_decrosstalk[dff_traces_post_decrosstalk.cell_roi_id == cell_roi_id].dff.values[0],
                   color='blue', label='with decrosstalk')
        ax[1].set_title('production dF/F trace without decrosstalk')

        ax[2].plot(dff_traces_pre_decrosstalk[dff_traces_pre_decrosstalk.cell_roi_id == cell_roi_id].dff.values[0],
                   color='black', label='without decrosstalk')
        ax[2].set_title('dev dF/F trace after decrosstalk')

        for i in range(3):
            ax[i].set_xlabel('2P frames')
            ax[i].set_ylabel('dF/F')

        fig.tight_layout()
        title = dataset.metadata_string
        plt.suptitle(title, x=0.53, y=1.02, fontsize=16)

        save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/decrosstalk_validation'
        utils.save_figure(fig, figsize, save_dir, 'dFF_before_and_after_decrosstalk_comparison',
                          title + '_' + str(cell_roi_id))
        plt.close()