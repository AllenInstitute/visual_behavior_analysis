import argparse

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.platform_paper_figures as ppf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_container_id", type=int,
                        help="Container ID to process")
    args = parser.parse_args()
    ophys_container_id = args.ophys_container_id
    print('ophys_container_id:', ophys_container_id)

    use_events = True
    filter_events = True

    folder = 'matched_cell_roi_and_trace_examples'
    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots/cell_matching'

    cells_table = loading.get_cell_table()
    cells_table = utilities.limit_to_last_familiar_second_novel_active(cells_table)
    cells_table = utilities.limit_to_containers_with_all_experience_levels(cells_table)

    for cell_specimen_id in cells_table[cells_table.ophys_container_id == ophys_container_id].cell_specimen_id.unique():
        try:
            # ppf.plot_matched_roi_and_trace(ophys_container_id, cell_specimen_id, limit_to_last_familiar_second_novel=True,
            #                                use_events=use_events, filter_events=filter_events, save_figure=True)
            cell_metadata = cells_table[cells_table.cell_specimen_id==cell_specimen_id]
            if len(cell_metadata) == 3:
                ppf.plot_matched_roi_and_traces_example(cell_metadata, include_omissions=True,
                                                        use_events=use_events, filter_events=filter_events, save_dir=save_dir, folder=folder)
                print('plot saved for', cell_specimen_id)
        except Exception as e:
            print('problem for', cell_specimen_id)
            print(e)


