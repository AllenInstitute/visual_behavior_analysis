import argparse

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.qc.single_cell_plots as scp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_container_id", type=int,
                        help="Container ID to process")
    args = parser.parse_args()
    ophys_container_id = args.ophys_container_id
    print('ophys_container_id:', ophys_container_id)

    # use_events = False
    # filter_events = False

    # folder = 'matched_cell_roi_and_trace_examples'
    # save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/cell_matching'

    save_dir = r'/allen/programs/mindscope/workgroups/learning/ophys/learning_project_cache'
    import pandas as pd
    experiments_table = pd.read_csv(os.path.join(save_dir, 'mFISH_project_expts.csv'))
    print(len(experiments_table))

    # get cells that are matched in all sessions
    matched_cells_df = utilities.get_matched_cells_for_learning_mFISH()
    # get just the matched cells for this container
    matched_cell_specimen_ids = matched_cells_df[matched_cells_df.ophys_container_id==ophys_container_id].cell_specimen_id.unique()

    for cell_specimen_id in matched_cell_specimen_ids:
        try:
            scp.plot_across_session_responses_from_dataset_dict(ophys_container_id, cell_specimen_id, experiments_table,
                                                            data_type='dff', save_figure=True)

            # ppf.plot_matched_roi_and_trace(ophys_container_id, cell_specimen_id, limit_to_last_familiar_second_novel=True,
            #                                use_events=use_events, filter_events=filter_events, save_figure=True)
            # cell_metadata = cells_table[cells_table.cell_specimen_id==cell_specimen_id]
            # if len(cell_metadata) == 3:
            #     ppf.plot_matched_roi_and_traces_example(cell_metadata, include_omissions=True,
            #                                             use_events=use_events, filter_events=filter_events, save_dir=save_dir, folder=folder)
            #     print('plot saved for', cell_specimen_id)
        except Exception as e:
            print('problem for', cell_specimen_id)
            print(e)


