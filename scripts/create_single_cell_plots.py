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
    save_dir = os.path.join(base_dir, glm_version)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # a folder within save_dir will be created called 'matched_cell_examples' for plots to go in
    # if you want to use an additional folder within save_dir/matched_cell_examples, name a sub_folder name here
    sub_folder = 'rois_traces_dropouts_weights_matched'
    # sub_folder = None
    data_type = 'filtered_events'

    # get experiments and cells tables limited to the data you want to plot
    # whatever filtering is applied here will be applied to GLM results
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=True, limit_to_closest_active=True)
    print(len(experiments_table), 'expts in expts table')
    cells_table = loading.get_cell_table(platform_paper_only=True, limit_to_closest_active=True,
                                         limit_to_matched_cells=False, add_extra_columns=True)
    print(len(cells_table.ophys_experiment_id.unique()), 'expts in cells table')
    print('should only be 402 experiments if limited to matched platform dataset')
    print(len(cells_table.cell_specimen_id.unique()), 'unique cell_specimen_ids in cells table')

    results_pivoted, weights_df, kernels = psc.get_GLM_outputs(glm_version, base_dir, folder, experiments_table, cells_table)
    print(len(results_pivoted.cell_specimen_id.unique()), 'unique cell_specimen_ids in results_pivoted')
    print(len(weights_df.cell_specimen_id.unique()), 'unique cell_specimen_ids in weights_df')


    # set features to use in plots
    dropout_features = ['all-images', 'omissions', 'behavioral', 'task']
    # dropout_features = ['variance_explained_full', 'all-images', 'omissions', 'behavioral', 'task']
    # features to use for weights_df
    weights_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'hits', 'misses', 'omissions']
    # weights_features = ['image0', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'hits', 'misses', 'omissions', 'running', 'pupil]

    for cell_specimen_id in cells_table[cells_table.ophys_container_id == ophys_container_id].cell_specimen_id.unique():
        try:
            print('generating plot for', cell_specimen_id)
            psc.plot_cell_rois_and_GLM_weights(cell_specimen_id, cells_table, experiments_table, results_pivoted, weights_df, dropout_features,
                                      weights_features, kernels, save_dir, sub_folder, data_type)
        except Exception as e:
            print('problem for', cell_specimen_id)
            print(e)

