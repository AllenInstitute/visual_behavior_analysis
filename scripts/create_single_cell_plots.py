import os
import numpy as np
import pandas as pd
import argparse

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as psc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_container_id", type=int,
                        help="Container ID to process")
    args = parser.parse_args()
    ophys_container_id = args.ophys_container_id
    print('ophys_container_id:', ophys_container_id)

    use_events = True
    filter_events = True

    ### first plot cell ROI and change and omission triggered averages across sessions
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
    base_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/figure_4'
    glm_version = '24_events_all_L2_optimize_by_session'

    # set data_type for cell response plots, can be 'dff', 'events', 'filtered_events'
    data_type = 'filtered_events'

    # set base_dir to load and save results
    base_dir = os.path.join(base_dir, glm_version)
    # folder in save_dir where you want to load GLM results from
    glm_output_folder = '220308'
    glm_output_folder = '220310_across_session_norm'
    glm_output_dir = os.path.join(base_dir, glm_output_folder)
    # glm_output_dir = None
    # if glm_output_dir is None, GLM results will be generated for the full dataset and nothing will be saved

    # if glm_output_dir is provided, plots will go into a folder within glm_output_dir called 'matched_cell_examples'
    # if no glm_output_dir, plots will go into a folder within base_dir called 'matched_cell_examples'
    if glm_output_dir:
        plot_save_dir = glm_output_dir
    else:
        plot_save_dir = base_dir

    # get experiments and cells tables limited to the data you want to plot
    # whatever filtering is applied here will be applied to GLM results
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=True, limit_to_closest_active=True)
    print(len(experiments_table), 'expts in expts table')
    cells_table = loading.get_cell_table(platform_paper_only=True, limit_to_closest_active=True,
                                         limit_to_matched_cells=True, add_extra_columns=True)
    print(len(cells_table.ophys_experiment_id.unique()), 'expts in cells table')
    print('should only be 402 experiments if limited to matched platform dataset')
    print(len(cells_table.cell_specimen_id.unique()), 'unique cell_specimen_ids in cells table')

    # get GLM output
    results_pivoted, weights_df, kernels = psc.get_GLM_outputs(glm_version, experiments_table, cells_table, glm_output_dir)
    print(len(results_pivoted.cell_specimen_id.unique()), 'unique cell_specimen_ids in results_pivoted')
    print(len(weights_df.cell_specimen_id.unique()), 'unique cell_specimen_ids in weights_df')

    # set features to use in plots
    dropout_features = ['all-images', 'omissions', 'behavioral', 'task']
    # dropout_features = ['variance_explained_full', 'all-images', 'omissions', 'behavioral', 'task']
    # features to use for weights_df
    weights_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'hits', 'misses', 'omissions']
    # weights_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'hits', 'misses', 'omissions', 'running', 'pupil]

    ### for putting all cells in one folder  ###
    # if you want to use an additional folder within plot_save_dir/matched_cell_examples, name a plot_sub_folder name here
    # plot_sub_folder = 'rois_traces_dropouts_weights_matched'
    # for cell_specimen_id in cells_table[cells_table.ophys_container_id == ophys_container_id].cell_specimen_id.unique():
    #     try:
    #         print('generating plot for', cell_specimen_id)
    #         psc.plot_cell_rois_and_GLM_weights(cell_specimen_id, cells_table, experiments_table, results_pivoted, weights_df, dropout_features,
    #                                   weights_features, kernels, plot_save_dir, plot_sub_folder, data_type)
    #     except Exception as e:
    #         print('problem for', cell_specimen_id)
    #         print(e)

    ### for putting cells in cluster specific folders  ###

    # load cluster IDs per cell
    file = [file for file in os.listdir(glm_output_dir) if 'cluster_labels' in file]
    cluster_ids = pd.read_hdf(os.path.join(glm_output_dir, file[0]), key='df')
    # add ophys_container from metadata
    cells_table = loading.get_cell_table()
    # get metadata for this container
    tmp = cluster_ids.merge(cells_table, on=['cre_line', 'cell_specimen_id'], how='right').drop_duplicates(subset='cell_specimen_id')
    container_data = tmp[tmp.ophys_container_id == ophys_container_id]


    ### ROIs, dropouts and weights
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
            psc.plot_cell_rois_and_GLM_weights(cell_specimen_id, cells_table, experiments_table,
                                               results_pivoted, weights_df, dropout_features,
                                               weights_features, kernels, save_dir, folder,
                                               data_type)
        except Exception as e:
            print('problem for', cell_specimen_id)
            print(e)


