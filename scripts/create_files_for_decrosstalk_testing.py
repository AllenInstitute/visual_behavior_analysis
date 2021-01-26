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


if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]

    dataset = loading.get_ophys_dataset(experiment_id, include_invalid_rois=True)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files=True)
    # save full set of traces
    dff_traces = dataset.dff_traces.copy()
    dff_traces.to_hdf(os.path.join(analysis.analysis_dir, 'dff_traces_post_decrosstalk.h5'), key='df')
    # identify NaN traces before response_df creation
    dff_traces = dataset.dff_traces.copy()
    non_nan_inds = [row for row in range(len(dff_traces)) if np.isnan(dff_traces.iloc[row].dff[0]) == False]
    # dataset.dff_traces = dff_traces.iloc[non_nan_inds]
    valid_cell_specimen_ids = dff_traces.iloc[non_nan_inds].index.values
    # get stimulus locked responses & overwrite in cache_dir to compare to non-decrosstalked examples
    sdf = analysis.get_response_df(df_name='stimulus_response_df')
    sdf = sdf[sdf.cell_specimen_id.isin(valid_cell_specimen_ids)]