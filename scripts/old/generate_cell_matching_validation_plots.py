import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.dataset.cell_matching_dataset import CellMatchingDataset
from visual_behavior.ophys.plotting import summary_figures as sf
from visual_behavior.ophys.container_analysis import utilities as ut


if __name__ == '__main__':
    # cache_dir = r'\\allen\programs\braintv\workgroups\ophysdev\OPhysCore\Analysis\2018-08 - Behavior Integration test'

    container_info = pd.read_csv(os.path.join(cache_dir,'cell_matching_results','container_info.csv'))
    container_id = container_info.container_id.values[3]
    lims_ids = ut.get_lims_ids_for_container(container_id)
    cell_matching_dataset_dict = ut.get_cell_matching_dataset_dict(lims_ids, cache_dir)

    reg_df = ut.get_registration_df(container_id, cell_matching_dataset_dict)
    ut.plot_registration_results(container_id, cell_matching_dataset_dict)
    matrix = ut.get_ssim_matrix(lims_ids, reg_df)
    ut.plot_ssim_matrix(matrix, container_id, lims_ids, cell_matching_dataset_dict, label='stim_names')
    ut.plot_ssim_matrix(matrix, container_id, lims_ids, cell_matching_dataset_dict, label=None)
    ut.plot_ssim_values(reg_df, container_id)

    ut.get_cell_matching_results_cell_specimen_ids(container_id)
    valid_matching_df = ut.get_valid_cell_matching_results_cell_specimen_ids(container_id, cell_matching_dataset_dict)
    counts = ut.get_matching_cell_counts(valid_matching_df)
    cdf = ut.get_cumulative_matching_cell_counts(counts)
    ut.plot_cumulative_matching_cell_counts(cdf, container_id)
    ut.plot_cumulative_matching_cell_counts(cdf, container_id, ymax=250)

    ut.plot_matched_cells_matrix(container_id, cell_matching_dataset_dict, label='stim_names')
    ut.plot_fraction_matched_cells_matrix(container_id, cell_matching_dataset_dict, label='stim_names')
    ut.plot_fraction_matched_cells_matrix(container_id, cell_matching_dataset_dict, label=None)

    ut.plot_cell_matching_validation(container_id, cell_matching_dataset_dict)






