import os
import numpy as np
import pandas as pd
import argparse

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as psc

from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering import plotting

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_container_id", type=int,
                        help="Container ID to process")
    args = parser.parse_args()
    ophys_container_id = args.ophys_container_id
    print('ophys_container_id:', ophys_container_id)

    # ophys_container_id = 791352433


    ### first plot cell ROI and change and omission triggered averages across sessions
    #
    # use_events = True
    # filter_events = True
    #
    # folder = 'matched_cell_roi_and_trace_examples'
    # save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/cell_matching'
    #
    # cells_table = loading.get_cell_table()
    # cells_table = utilities.limit_to_last_familiar_second_novel_active(cells_table)
    # cells_table = utilities.limit_to_containers_with_all_experience_levels(cells_table)
    # # dont limit to cells matched in all sessions for now
    #
    # for cell_specimen_id in cells_table[cells_table.ophys_container_id == ophys_container_id].cell_specimen_id.unique():
    #     try:
    #         # ppf.plot_matched_roi_and_trace(ophys_container_id, cell_specimen_id, limit_to_last_familiar_second_novel=True,
    #         #                                use_events=use_events, filter_events=filter_events, save_figure=True)
    #         cell_metadata = cells_table[cells_table.cell_specimen_id==cell_specimen_id]
    #         if len(cell_metadata) == 3:
    #             ppf.plot_matched_roi_and_traces_example(cell_metadata, include_omissions=True,
    #                                                     use_events=use_events, filter_events=filter_events, save_dir=save_dir, folder=folder)
    #             print('plot saved for', cell_specimen_id)
    #     except Exception as e:
    #         print('problem for', cell_specimen_id)
    #         print(e)


    ### now plot cell ROIs with GLM output ###

    import visual_behavior_glm.GLM_params as glm_params
    import visual_behavior_glm.GLM_analysis_tools as gat
    import visual_behavior_glm.GLM_visualization_tools as gvt

    # directory and folders to save plots to
    base_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4'
    glm_version = '24_events_all_L2_optimize_by_session'

    # set data_type for cell response plots, can be 'dff', 'events', 'filtered_events'
    data_type = 'events'

    # set base_dir to load and save results
    base_dir = os.path.join(base_dir, glm_version)

    # folder in save_dir where you want to load GLM results from
    glm_output_folder = '220420_SAC'
    glm_output_dir = os.path.join(base_dir, glm_output_folder)
    print(glm_output_dir)

    # directory to save plots to, in a subfolder called 'matched_cell_examples
    plot_save_dir = glm_output_dir

    # get experiments and cells tables limited to the data you want to plot
    # whatever filtering is applied here will be applied to GLM results
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=True, limit_to_closest_active=True)
    print(len(experiments_table), 'expts in expts table')
    cells_table = loading.get_cell_table(platform_paper_only=True, limit_to_closest_active=True,
                                         limit_to_matched_cells=True, add_extra_columns=True)
    print(len(cells_table.ophys_experiment_id.unique()), 'expts in cells table')
    print('should only be 402 experiments if limited to matched platform dataset')
    print(len(cells_table.cell_specimen_id.unique()), 'unique cell_specimen_ids in cells table')

    ### get GLM output ###
    # whatever pre-processing has been applied to results_pivoted saved to glm_output_dir will be applied here
    # ex: across session norm, signed weights, etc.
    print('loading GLM results')
    results_pivoted, weights_df, kernels = processing.load_GLM_outputs(glm_version, glm_output_dir)
    print(len(results_pivoted.cell_specimen_id.unique()), 'unique cell_specimen_ids in results_pivoted')
    print(len(weights_df.cell_specimen_id.unique()), 'unique cell_specimen_ids in weights_df')
    print('GLM results loaded')

    # # make sure weights and dropouts are limited to matched experiments / cells
    # matched_cells = cells_table.cell_specimen_id.unique()
    # matched_experiments = cells_table.ophys_experiment_id.unique()
    # weights_df = weights_df[weights_df.ophys_experiment_id.isin(matched_experiments)]
    # weights_df = weights_df[weights_df.cell_specimen_id.isin(matched_cells)]
    # results_pivoted = results_pivoted.reset_index() # reset just in case
    # results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(matched_experiments)]
    # results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]

    # set features to use in plots
    # dropout_features = ['variance_explained_full', 'all-images', 'omissions', 'behavioral', 'task']
    dropout_features = ['all-images', 'omissions', 'behavioral', 'task']

    # features to use for weights_df
    weights_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'hits', 'misses', 'omissions']
    # weights_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'hits', 'misses', 'omissions', 'running', 'pupil]

    # load cluster IDs per cell
    file = [file for file in os.listdir(glm_output_dir) if 'cluster_labels' in file]
    cluster_ids = pd.read_hdf(os.path.join(glm_output_dir, file[0]), key='df')
    # add ophys_container from metadata
    cells_table = loading.get_cell_table()
    # get metadata for this container
    tmp = cluster_ids.merge(cells_table, on=['cre_line', 'cell_specimen_id'], how='right').drop_duplicates(subset='cell_specimen_id')
    container_data = tmp[tmp.ophys_container_id == ophys_container_id]

    # make cre and cluster ID specific folders if they dont already exist
    cre_line = container_data.cre_line.unique()[0]
    save_dir = os.path.join(glm_output_dir, 'matched_cell_examples', cre_line)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # loop through cells for this container and add to cre line and cluster ID specific folders
    container_csids = container_data.cell_specimen_id.unique()
    for cell_specimen_id in container_csids:
        cluster_id = container_data[container_data.cell_specimen_id==cell_specimen_id].cluster_id.unique()[0]
        folder = 'cluster_' + str(int(cluster_id))
        try:
            print('generating plot for', cell_specimen_id)
            psc.plot_cell_rois_and_GLM_weights(cell_specimen_id, cells_table, experiments_table, dropout_features,
                                               results_pivoted, weights_df, weights_features, kernels,
                                               save_dir, folder, data_type)
        except Exception as e:
            print('problem for', cell_specimen_id)
            print(e)


